"""Rebuild the archived clinical aggregates with correct metric units."""
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/jbi_editorial_review_20260912/figures'
spec = importlib.util.spec_from_file_location('aligned', ROOT / 'scripts/build_aligned_clinical_real_results.py')
aligned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aligned)
aligned.Y_LABELS['cell_error'] = 'RMSE'
aligned.Y_LABELS['trajectory_error'] = 'RMSE'

def save(fig, stem):
    for ax in fig.axes:
        ax.set_xlabel('Late-cell deletion target', fontsize=18, fontweight='bold')
        ax.yaxis.label.set_size(18)
        ax.tick_params(labelsize=17)
        title = ax.get_title(loc='left')
        title = title.replace('Mean absolute coefficient bias', 'Mean absolute\ncoefficient bias')
        title = title.replace('Coefficient mean squared error', 'Coefficient\nmean squared error')
        title = title.replace('95% coefficient coverage', '95% coefficient\ncoverage')
        ax.set_title(title, loc='left', fontsize=20, pad=40)
        for text in list(ax.texts):
            if text.get_text().replace('.', '').isdigit():
                text.remove()
            else:
                text.set_fontsize(15)
        for line in ax.lines:
            if line.get_marker() not in ('None', None, ''):
                line.set_markersize(12)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(19)
    fig.subplots_adjust(bottom=.13, top=.76, hspace=.9, wspace=.75)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

aligned.save = save
for n, runs, folder, limits, number in [
    (500, 200, aligned.CLINICAL, aligned.CLINICAL_Y_LIMITS, 2),
    (2000, 100, aligned.CLINICAL_N2000, aligned.CLINICAL_N2000_Y_LIMITS, 3),
]:
    summary = aligned.clinical_summary(folder)
    aligned.build_clinical(summary, n_subjects=n, n_runs=runs,
                          stem=f'Figure_{number}_clinical_audited',
                          title=f'Clinical association benchmark (n={n:,})',
                          y_limits=limits)
