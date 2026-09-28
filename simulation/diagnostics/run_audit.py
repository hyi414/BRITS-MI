"""Reproducible editorial diagnostics; no published result files are overwritten."""
from pathlib import Path
import sys, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/jbi_editorial_review_20260912/diagnostics'
OUT.mkdir(exist_ok=True, parents=True)
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_neural_brits_mi_downstream_gradient as neural
from test_anchored_brits_mi_blend import blend_draws

started = time.time()
plan = {'purpose':'Post hoc component diagnostic, not a new confirmatory benchmark',
        'seeds':list(range(1200000,1200020)), 'n':500, 'Q':50, 'rates':[.2,.4,.6],
        'selection':'First 20 archived seeds in numeric order, same seeds and draws for every component',
        'components':['Raw recurrent draws','Forest anchor only','Anchor plus recurrent spread',
                      'Reported hybrid','Hybrid conditional mean only'],
        'training_repeated':False, 'tests':'No significance claim for post hoc ablation'}
(OUT/'diagnostic_plan.json').write_text(json.dumps(plan,indent=2))
coef, metrics, missing = [], [], []
for rate in plan['rates']:
    folder = ROOT/f'outputs/refined_brits_mi_assoc_n500_nsim100_m10_20_50_miss{round(rate*100)}_20260802'
    for seed in plan['seeds']:
        data = neural.generate_locked_dataset(seed,'baseline_harder',500,rate)
        cached = np.load(folder/f'raw_seed_{seed}.npz')
        raw = list(cached['raw_draws'])
        anchor = cached['anchor']
        assert len(raw)==50
        assert np.allclose(anchor[data['observed']],data['trajectory'][data['observed']])
        mean_weight = .2 if rate==.6 else .1
        cases = {
            'Raw recurrent draws':raw,
            'Forest anchor only':[anchor],
            'Anchor plus recurrent spread':blend_draws(raw,anchor,data['observed'],0,.7),
            'Reported hybrid':blend_draws(raw,anchor,data['observed'],mean_weight,.7),
            'Hybrid conditional mean only':[np.mean(blend_draws(raw,anchor,data['observed'],mean_weight,.7),axis=0)],
        }
        missing.append({'seed':seed,'rate':rate,'all_cell_missing':float(1-data['observed'].mean()),
                        'late_cell_missing':float(1-data['observed'][:,-3:,:].mean()),
                        'has_both_missing_and_observed_later':float(np.any((~data['observed'][:,:-1,:]) & data['observed'][:,1:,:],axis=(1,2)).mean())})
        for name, draws in cases.items():
            neural.METHOD_LABELS[name]=name
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rows, metric=neural.evaluate_completions(data,draws,seed,'baseline_harder',name,{})
                coef.extend(rows); metrics.append(metric)
            except Exception as e:
                metrics.append({'seed':seed,'target_missing':rate,'method':name,'failure':repr(e)})
        print(f'audit rate={rate:.1f} seed={seed} elapsed={time.time()-started:.1f}s',flush=True)
    pd.DataFrame(coef).to_csv(OUT/'component_coefficients.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'component_metrics.csv',index=False)
pd.DataFrame(missing).to_csv(OUT/'missingness_realization.csv',index=False)
c=pd.DataFrame(coef); m=pd.DataFrame(metrics)
summary=[]
for (rate,method),g in c.groupby(['target_missing','method'],sort=False):
    byterm=g.groupby('coefficient')['bias'].mean()
    byrun=g.groupby('seed').agg(coverage=('covered','mean'),mse=('squared_error','mean'))
    cell=m[(m.target_missing==rate)&(m.method==method)]
    summary.append({'rate':rate,'method':method,'runs':len(byrun),'terms':len(byterm),
                    'absolute_bias':byterm.abs().mean(),'coverage':byrun.coverage.mean(),
                    'coverage_mcse':byrun.coverage.std(ddof=1)/np.sqrt(len(byrun)),
                    'coefficient_mse':byrun.mse.mean(),
                    'cell_rmse':cell.imputation_rmse.mean(),
                    'summary_rmse':cell.trajectory_summary_rmse.mean()})
pd.DataFrame(summary).to_csv(OUT/'component_summary.csv',index=False)
(OUT/'component_runtime.json').write_text(json.dumps({'analysis_wall_seconds':time.time()-started,
    'note':'Cached completion reanalysis only, not neural-training time.'},indent=2))
print(pd.DataFrame(summary).round(4).to_string(index=False),flush=True)
