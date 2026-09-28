from pathlib import Path
from types import SimpleNamespace
import sys, json, warnings
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from longitudinal_sim.baselines import iterative_impute
OUT=ROOT/'output/jbi_editorial_review_20260912/diagnostics'
OUT.mkdir(exist_ok=True,parents=True)
rng=np.random.default_rng(20260912)
n,t,p=80,4,3
static=rng.normal(size=(n,2))
truth=2+static[:,0,None,None]+rng.normal(scale=.25,size=(n,t,p))
mask=(rng.uniform(size=truth.shape)>.35).astype(float)
mask[0,0,0]=1; truth[0,0,0]=0  # A genuine observed zero must stay observed.
obs=np.where(mask==1,truth,0)
data=SimpleNamespace(y_obs=obs,mask=mask,static=static,x=np.zeros((n,t,1)),lengths=np.full(n,t))
train=np.arange(60)
matrix=np.concatenate([obs.reshape(n,-1),mask.reshape(n,-1),static,data.x.reshape(n,-1)],axis=1)
with warnings.catch_warnings(record=True) as recorded:
    old=IterativeImputer(random_state=17,max_iter=15,sample_posterior=False).fit(matrix[train]).transform(matrix)[:,:t*p].reshape(n,t,p)
    corrected_matrix=matrix.copy()
    corrected_matrix[:,:t*p]=np.where(mask.reshape(n,-1)==1,obs.reshape(n,-1),np.nan)
    fixed=IterativeImputer(random_state=17,max_iter=15,sample_posterior=False,keep_empty_features=True).fit(corrected_matrix[train]).transform(corrected_matrix)[:,:t*p].reshape(n,t,p)
active=iterative_impute(data,train)
result={'n':n,'train_n':len(train),'seed':20260912,'artificial_missing_count':int((mask==0).sum()),
    'legacy_exactly_zero_fill':bool(np.array_equal(old,obs)),
    'legacy_missing_predictions_nonzero':int(np.count_nonzero(old[mask==0])),
    'corrected_missing_predictions_nonzero':int(np.count_nonzero(fixed[mask==0])),
    'corrected_observed_cells_unchanged':bool(np.array_equal(fixed[mask==1],truth[mask==1])),
    'observed_zero_preserved':bool(fixed[0,0,0]==0),
    'legacy_test_rmse':float(np.sqrt(np.mean((old[60:][mask[60:]==0]-truth[60:][mask[60:]==0])**2))),
    'corrected_test_rmse':float(np.sqrt(np.mean((fixed[60:][mask[60:]==0]-truth[60:][mask[60:]==0])**2))),
    'source_function_matches_legacy':bool(np.array_equal(active,obs)),
    'source_function_matches_corrected':bool(np.array_equal(active,fixed)),
    'source_function_observed_cells_fixed':bool(np.array_equal(active[mask==1],truth[mask==1])),
    'warnings':[str(w.message) for w in recorded],
    'scope':'Mechanism/unit diagnostic only, not a corrected image benchmark or package-default MICE result.'}
assert result['source_function_matches_corrected']
assert result['source_function_observed_cells_fixed']
(OUT/'image_imputer_encoding_after_fix.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
