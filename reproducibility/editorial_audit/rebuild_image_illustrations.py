"""Render illustrative simulator output, not a new performance benchmark."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from longitudinal_sim.config import ExperimentConfig
from longitudinal_sim.data import generate_dataset
from run_fast_image_fusion_nsim500 import image_features

OUT=ROOT/'output/jbi_editorial_review_20260912'
config=ExperimentConfig(n_subjects=300,seed=20260912,mask_rate=.5,
    dgp_profile='paper_britsadv_img065_altus',info_profile='balanced',n_classes=3,image_size=24)
data=generate_dataset(config)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':17,
    'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,'pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
labels=['F0-F1','F2','F3-F4']; rng=np.random.default_rng(20260912)
def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(3,4,figsize=(12,8.2))
fig.subplots_adjust(left=.13,right=.97,top=.98,bottom=.04,wspace=.08,hspace=.18)
chosen={}
for c in range(3):
    chosen[c]=rng.choice(np.flatnonzero(data.labels==c),4,replace=False).tolist()
    for j,i in enumerate(chosen[c]):
        ax=axes[c,j];ax.imshow(data.image[i,0],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    axes[c,0].set_ylabel(labels[c],fontsize=18,weight='bold',labelpad=14)
save(fig,'Figure_S6_ultrasound_examples')

fig,axes=plt.subplots(1,3,figsize=(12,4.2))
fig.subplots_adjust(left=.03,right=.97,top=.86,bottom=.04,wspace=.16)
for c,ax in enumerate(axes):
    ax.imshow(data.image[data.labels==c,0].mean(axis=0),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    ax.set_title(labels[c],fontweight='bold',pad=12);ax.axis('off')
save(fig,'Figure_S7_ultrasound_mean_images')

features=image_features(data.image)
panels=[(0,'Mean pixel intensity'),(1,'Pixel-intensity SD'),(2,'Mean gradient magnitude'),
        (7,'Center-minus-outer contrast'),(8,'SD of row means'),(11,'Left-minus-right contrast')]
colors=['#b7c6cf','#657e91','#254c67']
fig,axes=plt.subplots(2,3,figsize=(12,7.7))
fig.subplots_adjust(left=.08,right=.98,top=.93,bottom=.08,wspace=.44,hspace=.43)
for ax,(k,label),letter in zip(axes.ravel(),panels,'ABCDEF'):
    box=ax.boxplot([features[data.labels==c,k] for c in range(3)],tick_labels=labels,
        showfliers=False,patch_artist=True,widths=.55,medianprops={'color':'black','linewidth':1.8})
    for patch,color in zip(box['boxes'],colors):patch.set_facecolor(color)
    ax.set_title(f'{letter}  {label}',loc='left',fontsize=15,pad=12,weight='bold')
    ax.grid(axis='y',color='#dce0e2');ax.set_axisbelow(True)
save(fig,'Figure_S8_image_features')
(OUT/'diagnostics/image_illustration_manifest.json').write_text(json.dumps({
    'purpose':'Illustration only; not rerun benchmark estimates',
    'seed':20260912,'subjects':300,'profile':config.dgp_profile,'size':24,
    'selected_indices':chosen,'features':'Six selected pixel-derived entries of the actual 17-feature image design',
    'nodule_count_not_used':'Renderer nodule count is not a pixel-derived predictor in this feature extractor'},indent=2))
