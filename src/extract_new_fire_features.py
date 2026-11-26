"""Plan, submit, and collect new-fire feature exports without modifying original fires."""
import argparse
import json
import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from google.cloud import storage
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from src.ee_notebook_exports import (
    get_cropland_data, get_elevation_data, get_nbr_timeseries, export_table,
)

ORIGINAL = {'Camp', 'Camp Fire', 'Dixie', 'Dixie Fire', 'North Complex', 'August Complex'}
STATE = Path('data/new_fire_exports')
KEYS = ['fire_name', 'plot_id']


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def plan():
    data = pd.read_csv('data/phase_2_daily_clean.csv', parse_dates=['date'])
    data = data.loc[~data.fire_name.isin(ORIGINAL)]
    if data.groupby(KEYS)[['centroid_lat', 'centroid_lon']].nunique().max().max() != 1:
        raise ValueError('A plot has multiple coordinates.')
    STATE.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fire, rows in data.groupby('fire_name'):
        manifest[fire] = {
            'start': rows.date.min().strftime('%Y-%m-%d'),
            'end_exclusive': (rows.date.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            'plots': rows[KEYS + ['centroid_lat', 'centroid_lon']].drop_duplicates().to_dict('records'),
        }
    save(STATE / 'plan.json', manifest)
    print({fire: len(value['plots']) for fire, value in manifest.items()})
    return manifest


def resolve_geometry(fire, item):
    """Identify the MTBS event and verify the notebook's original grid construction."""
    points = ee.Geometry.MultiPoint([
        [plot['centroid_lon'], plot['centroid_lat']] for plot in item['plots']
    ])
    year = int(item['start'][:4])
    candidates = (ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
                  .filter(ee.Filter.eq('Incid_Name', fire.upper()))
                  .filter(ee.Filter.gte('Ig_Date', ee.Date(f'{year-1}-01-01').millis()))
                  .filter(ee.Filter.lt('Ig_Date', ee.Date(f'{year+1}-01-01').millis()))
                  .filterBounds(points.buffer(15000)))
    found = candidates.getInfo()['features']
    if len(found) != 1:
        raise ValueError(f'{fire}: found {len(found)} MTBS events. Review event metadata.')
    event = ee.Feature(found[0])
    ignition = pd.to_datetime(found[0]['properties']['Ig_Date'], unit='ms').strftime('%Y-%m-%d')
    coords = event.geometry().bounds().getInfo()['coordinates'][0]
    west, south = min(p[0] for p in coords), min(p[1] for p in coords)
    north = max(p[1] for p in coords)
    height = 9000 / 111000
    width = height / math.cos(math.radians((south + north) / 2))
    features = []
    for plot in item['plots']:
        lon, lat = plot['centroid_lon'], plot['centroid_lat']
        # Notebook centroids are cell centers on an unclipped perimeter-bounds grid.
        column = (lon - west) / width - .5
        row = (lat - south) / height - .5
        if max(abs(column - round(column)), abs(row - round(row))) > .01:
            raise ValueError(f"{fire}/{plot['plot_id']}: MTBS grid differs from CSV. Original polygon needed.")
        geometry = ee.Geometry.Rectangle(
            [lon-width/2, lat-height/2, lon+width/2, lat+height/2]
        )
        features.append(ee.Feature(geometry, plot))
    return ee.FeatureCollection(features), ignition


def daily_export(plots, fire, start, end, bucket, prefix):
    """One spatial mean per UTC date and plot, with SMAP first averaged within day."""
    era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(
        ['temperature_2m', 'total_precipitation_sum',
         'u_component_of_wind_10m', 'v_component_of_wind_10m'])
    smap = ee.ImageCollection('NASA/SMAP/SPL4SMGP/008').select(
        ['sm_surface', 'sm_rootzone', 'surface_temp'])
    count = ee.Date(end).difference(ee.Date(start), 'day').int()

    def extract(offset):
        date = ee.Date(start).advance(offset, 'day')
        next_date = date.advance(1, 'day')
        image = era.filterDate(date, next_date).mean().addBands(
            smap.filterDate(date, next_date).mean())
        reduced = image.reduceRegions(
            collection=plots, reducer=ee.Reducer.mean(), scale=9000, tileScale=4)
        return reduced.map(lambda feature: feature.set('date', date.format('YYYY-MM-dd')))

    rows = ee.FeatureCollection(ee.List.sequence(0, count.subtract(1)).map(extract)).flatten()
    task = export_table(
        collection=rows, description=f'features_daily_{fire.replace(" ", "_")}',
        bucket=bucket, fileNamePrefix=f'{prefix}/daily_{fire.replace(" ", "_")}',
        fileFormat='CSV')
    task.start()
    return task


def submit(args):
    ee.Initialize(project=args.project)
    manifest = json.loads((STATE / 'plan.json').read_text())
    state_file = STATE / ('tasks_drive.json' if args.destination == 'drive' else 'tasks.json')
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if args.destination == 'drive':
        drive = build('drive', 'v3', credentials=ee.data.get_persistent_credentials(), cache_discovery=False)
        folders = drive.files().list(
            q=f"name = '{args.drive_folder}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields='files(id,name)',
        ).execute()['files']
        if len(folders) > 1:
            raise ValueError('More than one export folder has the requested name.')
        if not folders:
            drive.files().create(body={'name': args.drive_folder, 'mimeType': 'application/vnd.google-apps.folder'},
                                 fields='id').execute()
        args.bucket = f'drive:{args.drive_folder}'
    else:
        storage_client = storage.Client(project=args.project, credentials=ee.data.get_persistent_credentials())
        storage_client.bucket(args.bucket).reload()
    # Resolve every fire before launching exports, so geometry problems are visible first.
    prepared = {fire: resolve_geometry(fire, item) for fire, item in manifest.items()}
    for fire, item in manifest.items():
        plots, ignition = prepared[fire]
        record = state.setdefault(fire, {
            'ignition': ignition, 'bucket': args.bucket, 'prefix': args.prefix, 'tasks': {},
        })
        if (record['bucket'], record['prefix']) != (args.bucket, args.prefix):
            raise ValueError('Resume requires the original bucket and prefix.')
        jobs = {
            'cropland': lambda: get_cropland_data(plots, fire, args.bucket, args.prefix),
            'elevation': lambda: get_elevation_data(plots, fire, args.bucket, args.prefix),
            'nbr': lambda: get_nbr_timeseries(
                plots, fire, (pd.Timestamp(ignition)-pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
                (pd.Timestamp(ignition)+pd.DateOffset(years=1)).strftime('%Y-%m-%d'),
                args.bucket, args.prefix),
            'daily': lambda: daily_export(
                plots, fire, item['start'], item['end_exclusive'], args.bucket, args.prefix),
        }
        for kind, launch in jobs.items():
            if kind in record['tasks']:
                continue
            task = launch()
            record['tasks'][kind] = task.id
            save(state_file, state)
            print(f'{fire} {kind}: {task.id}', flush=True)


def collect(args):
    ee.Initialize(project=args.project)
    state_file = STATE / ('tasks_drive.json' if args.destination == 'drive' else 'tasks.json')
    state = json.loads(state_file.read_text())
    credentials = ee.data.get_persistent_credentials()
    if args.destination == 'drive':
        drive = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        folders = drive.files().list(
            q=f"name = '{args.drive_folder}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields='files(id)',
        ).execute()['files']
        if len(folders) != 1:
            raise ValueError('Export folder is missing or ambiguous.')
        folder_id = folders[0]['id']
    else:
        client = storage.Client(project=args.project, credentials=credentials)
    task_ids = [task for record in state.values() for task in record['tasks'].values()]
    statuses = {row['id']: row for row in ee.data.getTaskStatus(task_ids)}
    pending = []
    all_frames = []
    severity_coverage = []
    for fire, record in state.items():
        frames = {}
        stem = fire.replace(' ', '_')
        names = {'daily': f'daily_{stem}', 'cropland': f'cropland_{stem}',
                 'elevation': f'elevation_data_{stem}', 'nbr': f'nbr_timeseries_{stem}'}
        for kind, task_id in record['tasks'].items():
            status = statuses[task_id]
            print(f'{fire} {kind}: {status["state"]}')
            if status['state'] != 'COMPLETED':
                if args.partial and status['state'] in ('READY', 'RUNNING'):
                    pending.append((fire, kind))
                    continue
                raise RuntimeError(f'Exports unfinished: {status.get("error_message", status["state"])}')
            path = STATE / f'{names[kind]}.csv'
            if not path.exists():
                temporary = path.with_suffix('.csv.part')
                if args.destination == 'drive':
                    filename = f'{record["prefix"]}/{names[kind]}.csv'.replace('/', '_')
                    files = drive.files().list(
                        q=f"'{folder_id}' in parents and name = '{filename}' and trashed = false",
                        fields='files(id,name)',
                    ).execute()['files']
                    if len(files) != 1:
                        raise ValueError(f'Export file missing or ambiguous: {filename}')
                    with temporary.open('wb') as handle:
                        downloader = MediaIoBaseDownload(handle, drive.files().get_media(fileId=files[0]['id']))
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                else:
                    client.bucket(record['bucket']).blob(f'{record["prefix"]}/{names[kind]}.csv').download_to_filename(temporary)
                temporary.replace(path)
            frames[kind] = pd.read_csv(path)
        if len(frames) != 4:
            continue
        daily = frames['daily'].drop(columns=['.geo', 'system:index'], errors='ignore')
        daily['date'] = pd.to_datetime(daily.date)
        if daily.duplicated(KEYS + ['date']).any():
            raise ValueError('Daily export contains duplicate keys.')
        nbr = frames['nbr']
        nbr['date'] = pd.to_datetime(nbr.date)
        ignition = pd.Timestamp(record['ignition'])
        pre = nbr[nbr.date < ignition].groupby(KEYS).NBR.mean()
        post = nbr[nbr.date >= ignition].groupby(KEYS).NBR.mean()
        severity = (pre-post).rename('dNBR').reset_index()
        counts = pd.concat([
            nbr[nbr.date < ignition].groupby(KEYS).NBR.count().rename('pre_observations'),
            nbr[nbr.date >= ignition].groupby(KEYS).NBR.count().rename('post_observations'),
        ], axis=1).fillna(0).reset_index()
        severity_coverage.append(counts)
        for kind in ['cropland', 'elevation']:
            cols = [c for c in frames[kind] if c.startswith('pct_') or c == 'elevation']
            daily = daily.merge(frames[kind][KEYS+cols], on=KEYS, how='left', validate='many_to_one')
        daily = daily.merge(severity, on=KEYS, how='left', validate='many_to_one')
        daily['days_post_fire'] = (daily.date-ignition).dt.days
        all_frames.append(daily)
    if pending:
        print(f'Downloaded completed exports; {len(pending)} tasks remain.')
        return False
    pd.concat(severity_coverage, ignore_index=True).to_csv(
        STATE / 'burn_severity_coverage.csv', index=False)
    pd.concat(all_frames, ignore_index=True).to_csv(STATE / 'new_fire_raw_features.csv', index=False)
    return True



def merge_model_features():
    """Rebuild notebook features for new fires and append the saved four-fire rows."""
    from src.data_loader import get_features

    labels = pd.read_csv('data/phase_2_daily_clean.csv', parse_dates=['date'])
    labels = labels.loc[~labels.fire_name.isin(ORIGINAL)]
    features = pd.read_csv(STATE / 'new_fire_raw_features.csv', parse_dates=['date'])
    daily_keys = KEYS + ['date']
    if features.duplicated(daily_keys).any():
        raise ValueError('New features have duplicate daily keys.')
    # Keep label-side satellite columns, and use exported weather to build model inputs.
    exported = ['temperature_2m', 'total_precipitation_sum', 'sm_surface', 'sm_rootzone',
                'surface_temp', 'dNBR', 'elevation', 'days_post_fire',
                'u_component_of_wind_10m', 'v_component_of_wind_10m']
    exported += [name for name in features if name.startswith('pct_')]
    weather = features[daily_keys + exported].rename(columns={
        'u_component_of_wind_10m': 'wind_u', 'v_component_of_wind_10m': 'wind_v',
    })
    data = labels.merge(weather, on=daily_keys, how='left', validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('Some label dates lack an exported feature record.')
    data.drop(columns='_merge', inplace=True)
    days = data.days_post_fire
    data['year'] = data.date.dt.year
    data['dNBR_x_days'] = data.dNBR * np.exp(-days / 30)
    data['moisture_index'] = data.sm_surface * (1 + data.total_precipitation_sum)
    data['temp_nox_proxy'] = np.maximum(data.temperature_2m - 308, 0)
    data['biomass_burn'] = data.pct_forest * data.dNBR
    data['urban_baseline'] = data.pct_urban * 1e-4
    data['wet_suppression'] = data.sm_surface + data.total_precipitation_sum
    data['recovery_index'] = data.sm_surface * days / 365
    data['wind_speed'] = np.hypot(data.wind_u, data.wind_v)
    data['wind_dispersion'] = data.wind_speed * days
    data['wind_direction'] = np.degrees(np.arctan2(data.wind_v, data.wind_u))
    data['calm_flag'] = (data.wind_speed < 2).astype(int)
    data['westerly_wind'] = -data.wind_u
    data['southerly_wind'] = data.wind_v
    data['early_post_fire'] = (days <= 30).astype(int)
    data['acute_phase'] = np.exp(-days / 15)
    data['smoke_concentration'] = data.dNBR * data.acute_phase / (data.wind_speed + .1)
    data['inversion_proxy'] = data.surface_temp / (data.wind_speed + .1)
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * data.day_of_year / 365.25
        data[f'day_of_year_sin_{harmonic}'] = np.sin(angle)
        data[f'day_of_year_cos_{harmonic}'] = np.cos(angle)
    existing = pd.read_csv('data/phase_2_model_clean.csv', parse_dates=['date'])
    required = get_features(existing)
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing model features: {missing}')
    save(STATE / 'feature_missingness.json', {
        name: int(data[name].isna().sum()) for name in required
    })
    # Existing four-fire rows are copied verbatim into this separate output.
    combined = pd.concat([existing, data], ignore_index=True)
    if combined.duplicated(daily_keys).any():
        raise ValueError('Combined model data contains duplicate records.')
    combined.to_csv('data/phase_2_all_fires_model.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'submit', 'collect'])
    parser.add_argument('--partial', action='store_true',
                        help='Cache completed downloads while other exports are still running.')
    parser.add_argument('--destination', choices=['gcs', 'drive'], default='gcs')
    parser.add_argument('--drive-folder', default='NOx_Phase2_New_Fire_Features_20261125')
    parser.add_argument('--project', default='sees-picker-2')
    parser.add_argument('--bucket', default='sees-bucket')
    parser.add_argument('--prefix', default='phase2_new_fire_features')
    args = parser.parse_args()
    if args.action == 'plan':
        plan()
    elif args.action == 'submit':
        submit(args)
    else:
        if collect(args):
            merge_model_features()

if __name__ == '__main__':
    main()
