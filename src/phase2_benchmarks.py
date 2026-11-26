import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.final_pinn import metrics

def main():
    d=pd.read_csv('data/phase_2_daily_clean.csv',parse_dates=['date'])
    d['wind_speed']=np.hypot(d.u_component_of_wind_10m,d.v_component_of_wind_10m)
    cols=['wind_speed']
    for k in (1,2,3):
     for name,fn in [('sin',np.sin),('cos',np.cos)]:
      c=f'{name}_{k}';d[c]=fn(2*np.pi*k*d.day_of_year/365.25);cols.append(c)
    result={}
    for fire in sorted(d.fire_name.unique()):
     outer=d[d.fire_name!=fire];test=d[d.fire_name==fire]
     train=outer[outer.groupby('fire_name').date.rank(method='dense',pct=True)<=.8]
     y=train.soil_nox_tau5hr;lo,hi=y.quantile([.01,.99])
     sc=StandardScaler().fit(train[cols])
     model=Ridge(alpha=1).fit(sc.transform(train[cols]),np.arcsinh(y.clip(lo,hi)))
     p=np.sinh(model.predict(sc.transform(test[cols])))
     result[fire]={'rows':len(test),'ridge':metrics(test.soil_nox_tau5hr.to_numpy(),p),'median':metrics(test.soil_nox_tau5hr.to_numpy(),np.full(len(test),y.median()))}
    Path('results/phase2/reduced_input_benchmarks.json').write_text(json.dumps(result,indent=2))
    print('Saved 16-fire benchmarks. Dates retained as supplied because ignition dates are absent for new fires.')


if __name__ == '__main__':
    main()
