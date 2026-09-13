"""Matched diagnostic of outcome availability, with no outcome-model tuning."""
from pathlib import Path
import sys,json,time,warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/jbi_editorial_review_20260912/diagnostics'
sys.path.insert(0,str(ROOT/'scripts'))
import run_package_default_association_comparators as pkg
import run_neural_brits_mi_downstream_gradient as neural
from package_default_imputation import run_package_imputer
from test_anchored_brits_mi_blend import blend_draws

plan={'purpose':'Post hoc matched outcome-availability diagnostic, not confirmatory',
      'n':500,'Q':10,'seeds':list(range(1200000,1200020)),'rates':[.2,.4,.6],
      'MICE':'R package defaults, maxit=5, same seeds and predictors; add observed outcome only',
      'BRITS':'Same archived neural draws and blend as primary benchmark, first 10 of 50 draws',
      'not_tested':'Substantive-model compatibility and passively updated interactions'}
(OUT/'outcome_mice_plan.json').write_text(json.dumps(plan,indent=2))
coefs=[];metrics=[];audits=[];start=time.time()
for rate in plan['rates']:
    for seed in plan['seeds']:
        data=pkg.generate_locked_dataset(seed,'baseline_harder',500,rate)
        verify=neural.generate_locked_dataset(seed,'baseline_harder',500,rate)
        assert np.array_equal(data['observed'],verify['observed'])
        frame=pd.concat([data['late_observed'].reset_index(drop=True),data['base'].reset_index(drop=True)],axis=1)
        for name,include in [('MICE without outcome',False),('MICE with outcome',True)]:
            local=frame.copy()
            if include:local['observed_fibrosis_outcome']=data['outcome']
            draws,meta=run_package_imputer(local,pkg.TARGET_COLUMNS,'mice',10,seed+720260)
            completions=[pkg.hist_sim._frame_to_late_trajectory(d,data['trajectory_obs']) for d in draws]
            neural.METHOD_LABELS[name]=name
            rows,metric=neural.evaluate_completions(verify,completions,seed,'baseline_harder',name,{})
            coefs.extend(rows);metrics.append(metric);audits.append({'rate':rate,'seed':seed,'method':name,**meta})
        folder=ROOT/f'outputs/refined_brits_mi_assoc_n500_nsim100_m10_20_50_miss{round(rate*100)}_20260802'
        cache=np.load(folder/f'raw_seed_{seed}.npz')
        completed=blend_draws(list(cache['raw_draws']),cache['anchor'],verify['observed'],.2 if rate==.6 else .1,.7)[:10]
        neural.METHOD_LABELS['BRITS-MI Q10']='BRITS-MI Q10'
        rows,metric=neural.evaluate_completions(verify,completed,seed,'baseline_harder','BRITS-MI Q10',{})
        coefs.extend(rows);metrics.append(metric)
        pd.DataFrame(coefs).to_csv(OUT/'outcome_mice_coefficients.csv',index=False)
        pd.DataFrame(metrics).to_csv(OUT/'outcome_mice_metrics.csv',index=False)
        pd.DataFrame(audits).to_csv(OUT/'outcome_mice_package_audit.csv',index=False)
        print(f'outcome diagnostic rate={rate} seed={seed}, elapsed={time.time()-start:.1f}s',flush=True)
c=pd.DataFrame(coefs);m=pd.DataFrame(metrics);summary=[]
for (rate,method),g in c.groupby(['target_missing','method']):
    run=g.groupby('seed').agg(coverage=('covered','mean'),mse=('squared_error','mean'))
    im=m[(m.target_missing==rate)&(m.method==method)]
    summary.append({'rate':rate,'method':method,'runs':len(run),'absolute_bias':g.groupby('coefficient').bias.mean().abs().mean(),
                    'coverage':run.coverage.mean(),'coverage_mcse':run.coverage.std(ddof=1)/np.sqrt(len(run)),
                    'coefficient_mse':run.mse.mean(),'cell_rmse':im.imputation_rmse.mean(),'summary_rmse':im.trajectory_summary_rmse.mean()})
pd.DataFrame(summary).to_csv(OUT/'outcome_mice_summary.csv',index=False)
print(pd.DataFrame(summary).round(4).to_string(index=False))
