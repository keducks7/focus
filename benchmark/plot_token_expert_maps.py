#!/usr/bin/env python3
"""Token-position x actual expert-ID maps from saved trajectories only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

RUNS = [('gsm8k_bs8_layer10_v1', 'GSM8K B8', 8),
        ('gsm8k_bs16_layer10_v1', 'GSM8K B16', 16),
        ('humaneval_bs8_layer10_v1', 'HumanEval B8', 8)]
LAYERS = (2, 10, 18)
STATUS_COLORS = ['#b6bdc7', '#ec793b', '#2783b5']


def read(path, step):
    with path.open() as f:
        meta = json.loads(next(f))
        recs = [r for line in f if (r := json.loads(line)).get('record_type') == 'token_trajectory_step'
                and r['step'] in (0, step)]
    return meta, recs


def matrix(tokens, layer, length, experts):
    z = np.full((length, experts), np.nan)
    status = np.zeros(length, dtype=int)
    routes = []
    for t in tokens:
        p = t['block_position']
        chosen = next(l for l in t['layers'] if l['layer_idx'] == layer)
        ids, weights = chosen['expert_ids'], chosen['router_weights']
        assert len(ids) == len(set(ids)) == len(weights) == 8
        z[p, ids] = weights
        status[p] = 1 if t['accepted'] else 2
        routes.append(set(ids))
    overlap = [len(a & b) / 8 for i, a in enumerate(routes) for b in routes[i+1:]]
    return z, status, float(np.mean(overlap)) if overlap else None


def draw(ax, z, status, cmap, norm, label):
    im = ax.imshow(z, origin='lower', aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                   extent=(-.5, z.shape[1]-.5, -.5, z.shape[0]-.5))
    for pos in np.flatnonzero(status == 0):
        ax.axhspan(pos-.5, pos+.5, color='#dfe3e8', linewidth=0)
    strip = ax.inset_axes([1.005, 0, .018, 1])
    strip.imshow(status[:, None], origin='lower', aspect='auto', interpolation='nearest',
                 cmap=ListedColormap(STATUS_COLORS), vmin=-.5, vmax=2.5)
    strip.set_axis_off()
    ax.set_title(label, fontsize=10)
    ax.set_xticks([0,32,64,96,128,160,192,224,255])
    ax.set_yticks([0,8,16,24,31])
    ax.tick_params(labelsize=8)
    ax.set_xlabel('Expert ID', fontsize=9)
    ax.set_ylabel('Token position', fontsize=9)
    return im


def finish(fig, axes, im, title, out):
    fig.suptitle(title, fontsize=14, y=.995)
    fig.colorbar(im, ax=axes, shrink=.72, pad=.035, label='Router weight (stored scaled value)')
    fig.legend(handles=[Patch(color=STATUS_COLORS[0], label='Accepted earlier: routing not recorded'),
                        Patch(color=STATUS_COLORS[1], label='Accepted this step'),
                        Patch(color=STATUS_COLORS[2], label='Still unresolved')],
               loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.get_layout_engine().set(rect=(0,.055,1,.855))
    for ext in ('png','pdf'):
        fig.savefig(out.with_suffix('.'+ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--step',type=int,default=4)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    loaded=[(run,title,batch,*read(args.input_root/run/f'token_trajectories_bs{batch}.jsonl',args.step))
            for run,title,batch in RUNS]
    vmax=max(w for _,_,_,_,recs in loaded for r in recs for t in r['tokens']
             for l in t['layers'] if l['layer_idx'] in LAYERS for w in l['router_weights'])
    norm=Normalize(0,vmax);cmap=plt.get_cmap('magma').copy();cmap.set_bad('#f0f0f0')
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    stats=[]
    for run,title,batch,meta,recs in loaded:
        folder=args.output_dir/run;folder.mkdir(exist_ok=True)
        for rec in recs:
            group,step=rec['group_id'],rec['step']
            # Initial-step baseline for group 0; chosen-step atlas covers every batch group.
            if step==0 and group!=0:continue
            requests=list(range(group*batch,(group+1)*batch))
            tokens={request:[t for t in rec['tokens'] if t['request_id']==request] for request in requests}
            maps={}
            for layer in LAYERS:
                fig,axes=plt.subplots(batch//4,4,figsize=(21,3.25*(batch//4)+1),layout='constrained',squeeze=False)
                for ax,request in zip(axes.flat,requests):
                    z,status,overlap=matrix(tokens[request],layer,meta['block_length'],meta['num_experts'])
                    maps[request,layer]=(z,status,overlap)
                    value='NA' if overlap is None else f'{overlap:.3f}'
                    im=draw(ax,z,status,cmap,norm,f'R{request:02d} | masked={len(tokens[request])} | Overlap@8={value}')
                    stats.append(dict(run=run,step=step,group=group,request=request,layer=layer,
                                      observed_tokens=len(tokens[request]),accepted_now=int((status==1).sum()),
                                      unresolved=int((status==2).sum()),accepted_earlier=int((status==0).sum()),
                                      mean_token_pair_overlap_at8='' if overlap is None else overlap))
                finish(fig,axes,im,f'{title} | step {step} | batch group {group} | layer {layer}\n'
                       'Token-level Top-8 routing; expert columns retain their original IDs',
                       folder/f'step{step}_group{group}_layer{layer}_all_requests')
            # Full-sized three-layer reference-style figure for each request in group 0.
            if step==args.step and group==0:
                for request in requests:
                    fig,axes=plt.subplots(3,1,figsize=(15,10),layout='constrained')
                    for ax,layer in zip(axes,LAYERS):
                        z,status,overlap=maps[request,layer]
                        value='NA' if overlap is None else f'{overlap:.3f}'
                        im=draw(ax,z,status,cmap,norm,f'Layer {layer} | mean token-pair Overlap@8={value}')
                    finish(fig,axes,im,f'{title} | step {step} | request R{request:02d} | block 0\n'
                           'Light background: expert not selected; full grey row: previously accepted / missing route',
                           folder/f'step{step}_request{request:02d}_three_layers')
            print(run,'step',step,'group',group,'done',flush=True)
    with (args.output_dir/'token_map_statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]),lineterminator='\n');writer.writeheader();writer.writerows(stats)
    (args.output_dir/'plot_metadata.json').write_text(json.dumps(dict(step=args.step,layers=LAYERS,
        color_min=0,color_max=vmax,weights='stored router_weights including routed_scaling_factor',
        expert_order='original ID, not sorted',token_pair_overlap='intersection size / 8; observed mask tokens only'),indent=2))


if __name__=='__main__':main()
