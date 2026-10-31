"""Behavior audit with explicit cold-item policy; leaves vendor sources untouched."""
import argparse, json, time, warnings, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from trecs.models import ImplicitMF
from trecs.models.recommender import BaseRecommender

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',message='No measurements are currently defined')
parser=argparse.ArgumentParser()
parser.add_argument('--iterations',type=int,default=5)
parser.add_argument('--corrected-only',action='store_true')
args=parser.parse_args()
OUT=Path(__file__).resolve().parents[1] / 'results' / 'behavior'
if args.corrected_only: OUT=OUT/f'corrected_als{args.iterations}'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mf_checked import CheckedImplicitMF
N,M,D,K=500,2000,20,10

def row_correlation(x,y):
    x=rankdata(x,axis=1); y=rankdata(y,axis=1)
    x-=x.mean(axis=1,keepdims=True);y-=y.mean(axis=1,keepdims=True)
    den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    ok=den>0
    return float(np.mean(np.sum(x[ok]*y[ok],axis=1)/den[ok]))

def run(seed,policy,update):
    started=time.perf_counter(); rng=np.random.default_rng(seed)
    users=rng.dirichlet(np.full(D,.5),size=N)
    items=rng.dirichlet(np.full(D,.5),size=M).T
    cls=ImplicitMF if policy=='original' else CheckedImplicitMF
    model=cls(num_users=N,num_items=M,num_latent_factors=K,
        actual_user_representation=users,actual_item_representation=items,
        num_items_per_iter=10,seed=seed,model_params={'iterations':args.iterations,'rng_spec':seed})
    true=users@items
    consumed=np.zeros((N,M),bool); known=np.zeros(M,bool)
    records=[]; snapshots=[]; choices=[]
    rows=np.arange(N)
    diagnostic_rng=np.random.default_rng(seed+10000)
    for t in range(120):
        if t in [20,21,40,80,119]:
            scores=model.predicted_user_item_scores
            # Draw eligible candidates per user for both rank correlation and randomized-user control.
            candidates=np.stack([diagnostic_rng.choice(np.flatnonzero(~consumed[u]),size=200,replace=False) for u in rows])
            ys=true[rows[:,None],candidates]
            xs=scores[rows[:,None],candidates]
            shuffled=scores[((rows+1)%N)[:,None],candidates]
            top=np.argpartition(xs,-10,axis=1)[:,-10:]
            control=np.argpartition(shuffled,-10,axis=1)[:,-10:]
            snapshots.append(dict(step=t,
                eligible_rank_correlation=row_correlation(xs,ys),
                sampled_top10_true_utility=float(np.take_along_axis(ys,top,axis=1).mean()),
                shuffled_user_top10_true_utility=float(np.take_along_axis(ys,control,axis=1).mean()),
                candidate_mean_utility=float(ys.mean()),
                known_items=int(known.sum())))
        model.run(1,startup=t<20,reset_interactions=False,repeated_items=False,no_new_items=True)
        c=np.asarray(model.interactions,dtype=int).copy()
        slate=np.asarray(model.items_shown,dtype=int)
        assert not consumed[rows,c].any()
        actual=true[rows,c]
        slate_utility=true[rows[:,None],slate]
        eligible=~consumed
        percentile=((true<=actual[:,None])&eligible).sum(axis=1)/eligible.sum(axis=1)
        records.append(dict(step=t,choice_utility=float(actual.mean()),
            slate_utility=float(slate_utility.mean()),choice_percentile=float(percentile.mean()),
            unfitted_slate_share=float((~known[slate]).mean()) if t>=20 else None))
        consumed[rows,c]=True;known[c]=True;choices.append(c)
        if t==19 or (t>=20 and update): model.train()
        if (t+1)%40==0: print(f'{seed} {policy} update={update}: {t+1}/120 ({time.perf_counter()-started:.1f}s)',flush=True)
    name=f'{seed}_{policy}_{"updated" if update else "frozen"}'
    result=dict(seed=seed,cold_policy=policy,retrained=update,seconds=round(time.perf_counter()-started,2),
        config=dict(users=N,items=M,actual_attributes=D,factors=K,als_iterations=args.iterations,startup_steps=20,evaluation_steps=100,drift=0),
        snapshots=snapshots,records=records)
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/f'{name}.npz',choices=np.stack(choices),users=users,items=items)
    print(f'FINISHED {name}: {result["seconds"]}s',flush=True)
    return result

all_results=[]
cases=[(123,'zero',True),(124,'zero',True),(123,'zero',False)] if args.corrected_only else [(123,'original',True),(123,'zero',True),(124,'zero',True),(123,'zero',False),(124,'zero',False)]
for seed,policy,update in cases:
    all_results.append(run(seed,policy,update))
(OUT/'summary.json').write_text(json.dumps(all_results,indent=2))
