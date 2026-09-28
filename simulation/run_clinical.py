"""Portable launcher for the reported recurrent-plus-forest clinical method."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,choices=[500,2000],required=True)
    p.add_argument('--runs',type=int,required=True)
    p.add_argument('--missing-rate',type=float,choices=[.2,.4,.6],required=True)
    p.add_argument('--mean-weight',type=float,required=True,
        help='Specify the archived configuration; do not infer a tuning setting')
    p.add_argument('--seed-start',type=int,default=1200000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Otherwise print the command only')
    a=p.parse_args()
    script=Path(__file__).parent/'clinical/scripts/run_refined_brits_mi_association.py'
    cmd=[sys.executable,str(script.resolve()),'--outdir',str(a.output.resolve()),
         '--n',str(a.n),'--nsim',str(a.runs),'--target-missing',str(a.missing_rate),
         '--mean-weight',str(a.mean_weight),'--noise-weight','0.70','--m-values','50',
         '--seed-start',str(a.seed_start),'--crossfit-folds','3','--hidden-size','48',
         '--max-epochs','45','--device','cpu']
    print(' '.join(cmd))
    if a.execute:subprocess.run(cmd,check=True)

if __name__=='__main__':main()
