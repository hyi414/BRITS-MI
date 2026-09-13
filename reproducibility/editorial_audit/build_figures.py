from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/jbi_editorial_review_20260912'
F=OUT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':18,
                     'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
                     'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold'})

def save(fig,name):
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(F/f'{name}.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

def axes_style(ax):
    ax.grid(axis='y',color='#dde1e3',lw=.8,zorder=0)
    ax.set_axisbelow(True)

folder=ROOT/'outputs/image_fusion_poster_locked_500runs_20260802'
pred=pd.read_csv(folder/'representative_subject_predictions.csv')
metrics=pd.read_csv(folder/'metrics_by_seed.csv')
names={'complete_fusion':'Full-data fusion','brits_joint_loss':'BRITS-MI-image',
       'missforest_fusion':'Random-forest surrogate','mean_impute_fusion':'Mean imputation','image_only':'Image only'}
colors={'complete_fusion':'#17364d','brits_joint_loss':'#008c86','missforest_fusion':'#df7a00',
        'mean_impute_fusion':'#7b8790','image_only':'#744eb3'}
fig,axs=plt.subplots(2,2,figsize=(16,14))
for j in range(2):
    axs[0,j].set_position([.08+j*.50,.60,.40,.33])
    axs[1,j].set_position([.08+j*.50,.08,.40,.25])
grid=np.linspace(0,1,300)
for key,name in names.items():
    g=pred[pred.method==key]; y=g.truth.values.astype(int);p=g[['prob_f0_f1','prob_f2','prob_f3_f4']].values
    yy=np.eye(3)[y]; curves=[];prs=[]
    for c in range(3):
        fpr,tpr,_=roc_curve(yy[:,c],p[:,c]);curves.append(np.interp(grid,fpr,tpr))
        precision,recall,_=precision_recall_curve(yy[:,c],p[:,c]);prs.append(np.interp(grid,recall[::-1],precision[::-1]))
    auc=roc_auc_score(y,p,multi_class='ovr',average='macro');ap=average_precision_score(yy,p,average='macro')
    axs[0,0].plot(grid,np.mean(curves,axis=0),color=colors[key],lw=2.4,label=f'{name}: {auc:.3f}')
    axs[0,1].plot(grid,np.mean(prs,axis=0),color=colors[key],lw=2.4,label=f'{name}: {ap:.3f}')
axs[0,0].plot([0,1],[0,1],color='#aab2b7',ls=':',lw=1)
for j,(title,xlabel,ylabel) in enumerate([('A  Fibrosis discrimination','False-positive rate','True-positive rate'),('B  Precision-recall performance','Recall','Precision')]):
    ax=axs[0,j];ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.01));axes_style(ax)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.20),fontsize=15,frameon=False,borderaxespad=0)
fig.text(.53,.399,'Across 500 matched runs: extension minus random-forest surrogate',ha='center',fontsize=15,fontweight='bold')
fig.text(.53,.375,'AUROC +0.00569 (95% CI 0.00418-0.00720); p < 0.001   |   AUPRC +0.00437; p = 0.002',ha='center',fontsize=14)
practical=['brits_joint_loss','missforest_fusion','mean_impute_fusion']
short=['Image-conditioned\nextension','Random-forest\nsurrogate','Mean\nimputation']
for j,(metric,title,ylabel) in enumerate([('imputation_rmse','C  Missing-cell recovery','Missing-cell RMSE'),('summary_mse','D  Trajectory-summary recovery','Trajectory-summary RMSE')]):
    ax=axs[1,j]; values=[metrics.loc[metrics.method==m,metric].dropna().values for m in practical]
    if metric=='summary_mse':values=[np.sqrt(x) for x in values]
    bp=ax.boxplot(values,positions=np.arange(3),widths=.48,patch_artist=True,showfliers=False,
                  medianprops={'color':'#19333d','linewidth':2},whiskerprops={'linewidth':1.4},capprops={'linewidth':1.4})
    for b,key in zip(bp['boxes'],practical):b.set(facecolor=colors[key],edgecolor=colors[key],alpha=.8)
    ax.set_xticks(np.arange(3),short);ax.set_title(title,loc='left',pad=27);ax.set_ylabel(ylabel);axes_style(ax)
    ax.text(0,1.03,'Lower is better',transform=ax.transAxes,fontsize=13,color='#4b5961')
save(fig,'Figure_4_image_extension_audited')

# Replot the real-data figure with a reference-relative label, not population bias.
real=pd.read_csv(ROOT/'outputs/real_brits_target_alignment_bootstrap10_validation100_20260802/all_method_comparison/all_method_metric_summary.csv')
methods=['BRITS-MI','MICE','Missforest','Mean imputation'];cn=['#008c86','#7451b9','#df7a00','#7b8790']
fig=plt.figure(figsize=(14,11));gs=fig.add_gridspec(2,6,left=.08,right=.98,top=.89,bottom=.08,hspace=.52,wspace=1.25)
axes=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6]),fig.add_subplot(gs[1,0:3]),fig.add_subplot(gs[1,3:6])]
panels=[('effect_abs_bias','A  Coefficient deviation','Mean absolute deviation'),('effect_mse','B  Coefficient squared error','Mean squared deviation'),
        ('effect_coverage','C  Reference inclusion','Interval inclusion proportion'),('masked_cell_standardized_rmse','D  Missing-cell recovery','Missing-cell NRMSE'),
        ('trajectory_summary_nrmse','E  Trajectory-summary recovery','Trajectory-summary NRMSE')]
rates=np.array([.2,.4,.6]);off=np.linspace(-.032,.032,4)
for ax,(metric,title,ylabel) in zip(axes,panels):
    for k,method in enumerate(methods):
        g=real[(real.method==method)&(real.metric==metric)].set_index('missing_rate').reindex(rates)
        ax.errorbar(rates+off[k],g['mean'],yerr=stats.t.ppf(.975,99)*g.mcse,fmt='o',ms=8,
                    elinewidth=1.8,capsize=4,color=cn[k],label=method.replace('Missforest','missForest'))
    ax.set_title(title,loc='left',pad=28,fontsize=16);ax.set_ylabel(ylabel)
    ax.text(0,1.035,'Source-cohort reference' if metric=='effect_coverage' else 'Lower is better',transform=ax.transAxes,fontsize=12.5)
    ax.set_xticks(rates,['20%','40%','60%']);ax.set_xlabel('Additional masking');axes_style(ax)
axes[0].set_ylim(.025,.112);axes[1].set_ylim(.003,.030);axes[2].set_ylim(.75,1.015)
axes[3].set_ylim(.46,.83);axes[4].set_ylim(.17,.66)
fig.legend(handles=[Line2D([0],[0],marker='o',lw=0,color=c,label=m.replace('Missforest','missForest'),markersize=9) for m,c in zip(methods,cn)],
           loc='upper center',bbox_to_anchor=(.5,.985),ncol=4,frameon=False,fontsize=16)
save(fig,'Figure_5_observed_cell_masking_audited')

# Supplementary component diagnostic, same archived draws without new training.
s=pd.read_csv(OUT/'diagnostics/component_summary.csv');c=pd.read_csv(OUT/'diagnostics/component_coefficients.csv')
mm=['Raw recurrent draws','Forest anchor only','Anchor plus recurrent spread','Reported hybrid','Hybrid conditional mean only']
cc=['#34699a','#90969b','#aa6b24','#008c86','#76539b']
fig,axs=plt.subplots(2,2,figsize=(13.5,10));fig.subplots_adjust(left=.09,right=.97,top=.84,bottom=.09,hspace=.48,wspace=.29)
for ax,(metric,title,ylabel) in zip(axs.flat,[('absolute_bias','A  Systematic coefficient bias','Mean absolute bias'),('coverage','B  Interval calibration','95% interval coverage'),
                                          ('cell_rmse','C  Cellwise reconstruction','Missing-cell RMSE'),('coefficient_mse','D  Coefficient accuracy','Coefficient MSE')]):
    for k,(method,col) in enumerate(zip(mm,cc)):
        g=s[s.method==method].set_index('rate').reindex(rates)
        ax.plot(rates+(k-2)*.014,g[metric],marker='o',markersize=8,lw=1.8,color=col,label=method)
        if metric=='coverage':ax.errorbar(rates+(k-2)*.014,g[metric],yerr=stats.t.ppf(.975,19)*g.coverage_mcse,fmt='none',capsize=4,color=col)
    ax.set_title(title,loc='left',pad=12,fontsize=20);ax.set_ylabel(ylabel,fontsize=18);ax.set_xticks(rates,['20%','40%','60%']);ax.set_xlabel('Late-cell deletion target',fontsize=18);ax.tick_params(labelsize=16);axes_style(ax)
    if metric=='coverage':ax.axhline(.95,color='#b2413a',ls='--',lw=1.2);ax.set_ylim(.75,1.02)
fig.legend(handles=[Line2D([0],[0],marker='o',lw=1.5,color=col,label=method) for method,col in zip(mm,cc)],loc='upper center',ncol=2,frameon=False,fontsize=15)
save(fig,'Figure_S9_component_diagnostic')
