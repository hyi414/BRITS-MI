from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/jbi_editorial_review_20260912/diagnostics'
sys.path.insert(0,str(ROOT/'scripts'))
import run_neural_brits_mi_downstream_gradient as neural

torch.manual_seed(20260912)
model=neural.NeuralBRITSMI(6,3,3,8).double()
n=6
y=torch.randn(n,6,3,dtype=torch.double)
mask=(torch.rand_like(y)>.35).double()
masked=torch.where(mask.bool(),y,torch.zeros_like(y))
times=torch.tensor([0,.07,.22,.46,.73,1],dtype=torch.double).repeat(n,1)
static=torch.randint(0,2,(n,3)).double()
epsilon=torch.randn_like(y)
label=torch.tensor([0,1,0,1,1,0],dtype=torch.double)
out=model(masked,mask,times,static,epsilon)
loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],label)
loss.backward()
groups={name:float(sum(p.grad.square().sum().item() for p in module.parameters() if p.grad is not None)**.5)
        for name,module in [('forward_RITS_GRU',model.forward_rits),('backward_RITS_GRU',model.backward_rits),
                           ('sequence_mean',model.global_mean),('sequence_scale',model.global_scale),
                           ('reconciliation_gate',model.combine_gate)]}
param=model.global_scale.weight
index=np.unravel_index(int(param.grad.abs().argmax()),param.shape)
analytic=float(param.grad[index]);old=float(param[index].detach());h=1e-5
values=[]
for delta in [h,-h]:
    with torch.no_grad(): param[index]=old+delta
    values.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(masked,mask,times,static,epsilon)['logits'],label).detach()))
with torch.no_grad():param[index]=old
numeric=(values[0]-values[1])/(2*h)
result={'parameter_count':sum(p.numel() for p in model.parameters()),'hidden_size_for_unit_test':8,
        'gradient_norms':groups,'analytic_gradient':analytic,'finite_difference_gradient':numeric,
        'relative_error':abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-10),
        'observed_cells_exact':bool(torch.equal(out['completed'][mask.bool()],y[mask.bool()])),
        'same_epsilon_reproducible':bool(torch.equal(out['completed'],model(masked,mask,times,static,epsilon)['completed'])),
        'different_epsilon_changes_missing':bool(not torch.equal(out['completed'][~mask.bool()],model(masked,mask,times,static,-epsilon)['completed'][~mask.bool()])),
        'full_clinical_parameter_count':sum(p.numel() for p in neural.NeuralBRITSMI(6,3,3,48).parameters())}
assert result['relative_error']<1e-5
assert all(groups.values()) and result['observed_cells_exact']
(OUT/'gradient_and_completion_tests.json').write_text(json.dumps(result,indent=2))

image=pd.read_csv(ROOT/'outputs/image_fusion_poster_locked_500runs_20260802/metrics_by_seed.csv')
columns=['auroc','auprc','log_loss','imputation_rmse','summary_mse']
a=image[image.method=='iterative_mice_fusion'].set_index('seed')[columns]
b=image[image.method=='zero_fill_fusion'].set_index('seed')[columns]
audit={'image_invalid_MICE_equal_zero_fill_all_metrics_all_runs':bool(np.array_equal(a.values,b.values)),
       'image_seed_counts':image.groupby('method').seed.nunique().to_dict()}
tests=[]
for metric in ['auroc','auprc']:
    pv=image.pivot(index='seed',columns='method',values=metric)
    d=pv.brits_joint_loss-pv.missforest_fusion
    se=d.std(ddof=1)/np.sqrt(len(d)); ci=stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=se)
    tests.append({'metric':metric,'n':len(d),'difference':d.mean(),'mcse':se,'ci_low':ci[0],'ci_high':ci[1],
                  'p_two_sided_unadjusted':stats.ttest_1samp(d,0).pvalue})
pd.DataFrame(tests).to_csv(OUT/'image_paired_tests_rechecked.csv',index=False)
c=pd.read_csv(ROOT/'outputs/clinical_association_n500_nsim200_final_20260802/coefficient_by_run.csv')
audit['clinical_seed_counts']={f'{r:.1f} {m}':int(g.seed.nunique()) for (r,m),g in c.groupby(['target_missing','method_label'])}
summary=[]
for (rate,method),g in c.groupby(['target_missing','method_label']):
    if method=='Complete case':continue
    byrun=g.groupby('seed').agg(coverage=('covered','mean'),mse=('squared_error','mean'))
    summary.append({'rate':rate,'method':method,'n':len(byrun),'terms':g.coefficient.nunique(),
                    'absolute_bias':g.groupby('coefficient').bias.mean().abs().mean(),
                    'coverage':byrun.coverage.mean(),'coverage_mcse':byrun.coverage.std(ddof=1)/np.sqrt(len(byrun)),
                    'coefficient_mse':byrun.mse.mean()})
pd.DataFrame(summary).to_csv(OUT/'clinical_aggregate_rechecked.csv',index=False)
audit['clinical_rates_share_same_seeds']=all(set(g.seed)==set(c[c.target_missing==.2].seed) for _,g in c.groupby('target_missing'))
(OUT/'provenance_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(result,indent=2));print(json.dumps(audit,indent=2));print(pd.DataFrame(tests).to_string(index=False))
