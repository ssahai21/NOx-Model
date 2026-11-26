import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.data_loader import get_features
from src.baseline_nn import BaselineNN
from src.final_pinn import metrics

def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description='Score saved weights against updated labels.')
    parser.add_argument('--source', type=Path, default=Path('results/logo_acute_weight_3'))
    parser.add_argument('--output', type=Path, default=Path('results/phase2/frozen_models'))
    args = parser.parse_args()
    source = args.source
    report=json.loads((source/'metrics.json').read_text())
    old=pd.read_csv('data/nox_clean.csv',parse_dates=['date'])
    old=old[old.days_post_fire>=0]
    new=pd.read_csv('data/phase_2_model_clean.csv',parse_dates=['date'])
    new=new[new.days_post_fire>=0]
    features=get_features(old)
    keys=['fire_name','plot_id','date']
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    results={}
    for fold in report['folds']:
     fire=fold['fire']
     outer=old[old.fire_name!=fire]
     train=outer[outer.groupby('fire_name').date.rank(method='dense',pct=True)<=.8]
     x=train[features].replace([np.inf,-np.inf],np.nan)
     med=x.median().fillna(0);sc=StandardScaler().fit(x.fillna(med))
     test=new[new.fire_name==fire].copy()
     filled=test[features].replace([np.inf,-np.inf],np.nan).fillna(med)
     scaled=np.clip(sc.transform(filled),-5,5)
     native=scaled.copy()
     land=[i for i,n in enumerate(features) if n.startswith('pct_')]
     native[:,land]=filled.iloc[:,land]
     inputs={'native':native,'standardized':scaled}
     p=test[keys+['soil_nox_tau5hr']].copy();results[fire]={}
     for name,model_info in fold['models'].items():
      cols=[]
      for seed,config in model_info['seeds'].items():
       model=BaselineNN(len(features))
       model.load_state_dict(torch.load(source/f'{fire}_{name}_{seed}.pt',map_location='cpu',weights_only=True))
       model.eval()
       with torch.inference_mode():
        pred=model(torch.tensor(inputs[config['land_cover_mode']],dtype=torch.float32)).flatten().numpy()
       c=f'{name}_{seed}';p[c]=np.sinh(pred.astype(float));cols.append(c)
      p[name+'_ensemble']=p[cols].mean(axis=1)
      results[fire][name]=metrics(p.soil_nox_tau5hr.to_numpy(),p[name+'_ensemble'].to_numpy())
     p.to_csv(out/f'{fire}_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
