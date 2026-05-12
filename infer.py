import os, sys, yaml, argparse, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from scipy.signal import resample
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from model.unet import ECGRecoverUNetV2

def load_config(p):
    with open(p) as f: return yaml.safe_load(f)

def clear_dir(p):
    p = Path(p)
    if p.exists():
        for f in p.glob('*'): f.is_file() and f.unlink()
    else: p.mkdir(parents=True, exist_ok=True)

def find_ckpt(d):
    p = Path(d)
    if not p.exists(): return None
    best = list(p.glob("*best*.pt"))
    return best[0] if best else next(p.glob("*.pt"), None)

def preprocess(df, cfg):
    cols = [c.strip() for c in df.columns]
    dc = df.columns[1:] if cols[0] in ['#', 'Time', 'time', ''] else df.columns
    leads = cfg['processing']['standard_leads']
    raw = np.full((12, len(df)), np.nan, dtype=np.float64)
    for c in dc:
        n = c.strip().upper().replace(' ', '')
        if n in leads: raw[leads.index(n)] = pd.to_numeric(df[c], errors='coerce').values
    L = cfg['processing']['target_length']
    sig = np.array([resample(raw[i], L).astype(np.float32) for i in range(12)])
    mask = np.zeros_like(sig)
    norm = np.zeros_like(sig)
    stats = []
    for i in range(12):
        v = ~np.isnan(sig[i])
        if v.sum() > 20:
            m, s = sig[i][v].mean(), sig[i][v].std() + 1e-8
            norm[i][v] = (sig[i][v] - m) / s
            mask[i] = v.astype(np.float32)
            stats.append((m, s))
        else: stats.append((0.0, 1.0))
    return np.nan_to_num(norm, nan=0.0), mask, stats

def run_inf(part, mask, ckpt, dev):
    m = ECGRecoverUNetV2(n_leads=12, base_channels=64)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True)['model_state_dict'])
    m.to(dev).eval()
    with torch.no_grad():
        return m(torch.from_numpy(part).float().unsqueeze(0).to(dev),
                 torch.from_numpy(mask).float().unsqueeze(0).to(dev)).squeeze(0).cpu().numpy()

def save_res(out, stats, leads, p, fs):
    res = np.array([out[i]*s+m for i,(m,s) in enumerate(stats)])
    pd.DataFrame({'#': np.arange(out.shape[1])/fs, **{l: res[i] for i,l in enumerate(leads)}}).to_csv(p, index=False, float_format='%.6f')

def save_plot(out, stats, leads, p, fs, dpi):
    res = np.array([out[i]*stats[i][1]+stats[i][0] for i in range(12)])
    t = np.arange(out.shape[1])/fs
    fig, ax = plt.subplots(4, 3, figsize=(10, 7), gridspec_kw={'hspace': 0.05})
    for a, n in zip(ax.flatten(), leads):
        idx = leads.index(n)
        a.plot(t, res[idx], color='#111', lw=0.8)
        a.set_title(n, fontsize=8, pad=2)
        a.set_xticks([]), a.set_yticks([]), a.grid(True, c='#E5E5E5', lw=0.4), a.spines[:].set_visible(False)
    fig.patch.set_facecolor('white')
    fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--config', default='config.yaml')
    pa.add_argument('--device', default=None)
    args = pa.parse_args()
    cfg = load_config(args.config)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    clear_dir(cfg['paths']['output_dir'])
    ckpt = find_ckpt(cfg['paths']['weights_dir'])
    if not ckpt: raise FileNotFoundError("No checkpoint in weights_dir")
    print(f"Checkpoint: {ckpt.name}")
    in_p, out_p = Path(cfg['paths']['input_dir']), Path(cfg['paths']['output_dir'])
    files = list(in_p.glob('*.csv'))
    if not files: raise FileNotFoundError("No CSV in input_dir")
    print(f"Processing {len(files)} files...")
    for f in files:
        try:
            p, m, s = preprocess(pd.read_csv(f), cfg)
            o = run_inf(p, m, ckpt, dev)
            if cfg['output']['save_csv']: save_res(o, s, cfg['processing']['standard_leads'], out_p/f"{f.stem}.csv", cfg['processing']['target_fs'])
            if cfg['output']['save_plots']: save_plot(o, s, cfg['processing']['standard_leads'], out_p/f"{f.stem}.png", cfg['processing']['target_fs'], cfg['output']['plot_dpi'])
            print(f"{f.name}")
        except Exception as e: print(f"{f.name}: {e}")
    print("Done.")

if __name__ == '__main__': main()