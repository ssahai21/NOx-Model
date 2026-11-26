"""Extraction functions adapted from SEESProject.ipynb cells 31, 35, and 27."""
import ee

def export_table(collection, description, bucket, fileNamePrefix, fileFormat='CSV'):
    if bucket.startswith('drive:'):
        return ee.batch.Export.table.toDrive(
            collection=collection, description=description, folder=bucket[6:],
            fileNamePrefix=fileNamePrefix.replace('/', '_'), fileFormat=fileFormat,
        )
    return ee.batch.Export.table.toCloudStorage(
        collection=collection, description=description, bucket=bucket,
        fileNamePrefix=fileNamePrefix, fileFormat=fileFormat,
    )


def get_cropland_data(plots, fire_name, bucket, export_prefix, scale=30):

    # USDA Cropland Data Layer
    cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filterDate('2019-01-01', '2020-01-01') \
            .first() \
            .select('cropland')

    # The crop classes you care about for wildfire analysis
    crop_classes = {
    # Natural vegetation — most relevant for wildfire NO2
    'forest':        [63, 141, 142, 143],
    'shrubland':     [64, 152],
    'grassland':     [176, 59, 60, 58],
    'wetland':       [87, 190, 195],
    'barren':        [65, 131],

    # Developed
    'urban':         [82, 121, 122, 123, 124],
    'water':         [83, 111, 112],

    # Agriculture — row crops
    'grain':         [1, 4, 21, 22, 23, 24, 25, 27, 28, 29, 30, 3],
    'oilseed':       [6, 31, 32, 33, 34, 35, 38],
    'legume':        [5, 10, 51, 52, 53, 42],
    'fiber':         [2],
    'hay_alfalfa':   [36, 37],
    'root_veg':      [41, 43, 46, 206, 247, 246],
    'vegetables':    [47, 49, 50, 54, 48, 209, 213, 214, 216, 219, 222, 227, 229, 243, 244, 245, 248, 249],
    'fruit_berry':   [55, 66, 67, 68, 69, 77, 221, 242, 250],
    'tree_nut':      [70, 71, 74, 75, 76, 204, 210, 211, 212, 215, 217, 218, 220, 223],
    'citrus':        [72],
    'vineyard':      [69],
    'double_crop':   [26, 225, 226, 228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 254],
    'fallow':        [61],
    'other':         [0, 11, 12, 13, 14, 44, 45, 56, 57, 88, 92, 205, 224, 81]
    }

    def extract_plot(plot):
        total_pixels = cdl.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=plot.geometry(),
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).get('cropland')

        props = {}
        for class_name, codes in crop_classes.items():
            # Create a binary mask for this class
            mask = cdl.remap(
                codes,
                [1] * len(codes),
                0
            )
            count = mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=plot.geometry(),
                scale=scale,
                maxPixels=1e9,
                bestEffort=True
            ).get('remapped')

            # Store as percentage
            props[f'pct_{class_name}'] = ee.Number(count).divide(ee.Number(total_pixels))

        return plot.set(props)

    plots_with_stats = plots.map(extract_plot)

    task = export_table(
        collection=plots_with_stats,
        description=f'cropland_{fire_name.replace(" ", "_")}',
        bucket=bucket,
        fileNamePrefix=f'{export_prefix}/cropland_{fire_name.replace(" ", "_")}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Submitted cropland for {fire_name} | Task ID: {task.id}")
    return task

def get_elevation_data(plots, fire_name, bucket, export_prefix, scale=30):

    image = ee.Image('NASA/NASADEM_HGT/001').select('elevation')

    def extract_plot(plot):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=plot.geometry(),
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4
        )
        return plot.set(stats)

    plots_with_stats = plots.map(extract_plot)

    task = export_table(
        collection=plots_with_stats,
        description=f'elevation_data_{fire_name.replace(" ", "_")}',
        bucket=bucket,
        fileNamePrefix=f'{export_prefix}/elevation_data_{fire_name.replace(" ", "_")}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Submitted elevation data for {fire_name} | Task ID: {task.id}")
    return task

def get_nbr_timeseries(plots, fire_name, start_date, end_date, bucket, export_prefix, scale=10):
    """
    Extracts NBR time series per plot using Sentinel-2.
    NBR = (NIR - SWIR) / (NIR + SWIR) = (B8 - B12) / (B8 + B12)
    """

    def mask_s2_clouds(image):
        time_start = image.get('system:time_start')
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000).set('system:time_start', time_start)

    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(ee.FeatureCollection(plots).geometry()) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(mask_s2_clouds)

    # Compute NBR as a band on each image
    def add_nbr(image):
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        return image.addBands(nbr).set('system:time_start', image.get('system:time_start'))

    s2_nbr = s2.map(add_nbr)

    # For each image, extract NBR per plot
    def extract_image(image):
        date = image.date().format('YYYY-MM-dd')

        def extract_plot(plot):
            stats = image.select('NBR').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=plot.geometry(),
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4
            )
            return plot.set(stats).set('date', date)

        return plots.map(extract_plot)

    timeseries = s2_nbr.map(extract_image).flatten()

    task = export_table(
        collection=timeseries,
        description=f'nbr_timeseries_{fire_name.replace(" ", "_")}',
        bucket=bucket,
        fileNamePrefix=f'{export_prefix}/nbr_timeseries_{fire_name.replace(" ", "_")}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Submitted NBR timeseries for {fire_name} | Task ID: {task.id}")
    return task
