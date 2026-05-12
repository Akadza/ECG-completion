import os, sys, yaml, time, gc, random, numpy as np, pandas as pd, wfdb, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from model.unet import ECGRecoverUNetV2

def load_cfg(p):
    with open(p) as f: return yaml.safe_load(f)

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def bandpass(sig, lo, hi, fs, o=4):
    nyq = fs/2; b, a = butter(o, [lo/nyq, hi/nyq], btype='band'); return filtfilt(b, a, sig, axis=-1)

def z_norm(sig):
    m = sig.mean(axis=-1, keepdims=True); s = sig.std(axis=-1, keepdims=True) + 1e-8
    return (sig - m) / s

def load_wfdb(p, fs, tl):
    try:
        r = wfdb.rdrecord(str(p)); sig = r.p_signal.T
        if sig.shape[1] != tl: sig = resample(sig, tl, axis=1)
        sig = z_norm(bandpass(sig, 0.5, 40, fs))
        return None if np.any(np.isnan(sig)) or np.any(np.isinf(sig)) else sig.astype(np.float32)
    except: return None

def load_mat(p, fs, tl):
    import scipy.io
    try:
        d = scipy.io.loadmat(str(p).replace('.hea','').replace('.mat','')+'.mat')
        sig = d.get('val', d.get('ecg'))
        if sig is None or sig.ndim!=2: return None
        if sig.shape[1]==12: sig=sig.T
        if sig.shape[0]<12: return None
        sig = sig[:12].astype(np.float64)
        rng = sig.max()-sig.min()
        if rng>100: sig/=4880.0
        elif rng<0.01: return None
        if sig.shape[1]!=tl: sig=resample(sig, tl, axis=1)
        sig = z_norm(bandpass(sig, 0.5, 40, fs))
        return None if np.any(np.isnan(sig)) or np.any(np.isinf(sig)) else sig.astype(np.float32)
    except: return None

def prepare_data(cfg):
    paths, proc = cfg['paths'], cfg['processing']
    ptb = Path(paths['ptbxl_path']); df = pd.read_csv(ptb/'ptbxl_database.csv')
    fc = 'filename_hr' if 'filename_hr' in df.columns else 'filename_lr'
    df['p'] = df[fc].apply(lambda x: ptb/str(x).replace('.dat','').replace('.hea',''))
    tr, vl = (df[df['strat_fold']<=8].reset_index(drop=True), df[df['strat_fold']==9].reset_index(drop=True)) if 'strat_fold' in df.columns else train_test_split(df, test_size=0.1)
    
    def load_split(df_s, nm):
        s, e = [], 0
        for _, r in tqdm(df_s.iterrows(), total=len(df_s), desc=nm, leave=False, ncols=70):
            x = load_wfdb(r['p'], proc['target_fs'], proc['target_length'])
            if x is not None: s.append(x)
            else: e += 1
        return np.array(s, dtype=np.float32) if s else None, e

    tr_d, e1 = load_split(tr, 'PTB-tr'); vl_d, e2 = load_split(vl, 'PTB-vl')
    print(f"PTB: tr={len(tr_d)}, vl={len(vl_d)}")
    
    ext_tr, ext_vl = [], []
    for p_key, loader in [('georgia_path', load_mat), ('china_path', load_wfdb)]:
        p = Path(paths[p_key])
        if p.exists() and p_key:
            fs = list(p.rglob('*.hea'))
            if fs:
                print(f"{p_key.split('_')[0]}: {len(fs)}")
                s = [x for x in tqdm((loader(f, proc['target_fs'], proc['target_length']) for f in fs), total=len(fs)) if x is not None]
                print(f"{len(s)} loaded")
                t, v = train_test_split(s, test_size=0.1)
                ext_tr.extend(t); ext_vl.extend(v)
    
    tr_arr = np.concatenate([tr_d] + ([np.array(ext_tr)] if ext_tr else []))
    vl_arr = np.concatenate([vl_d] + ([np.array(ext_vl)] if ext_vl else []))
    del tr_d, vl_d; gc.collect()
    return tr_arr, vl_arr

def get_mask(cfg):
    types, probs = cfg['training']['masks'], cfg['training']['mask_probs']
    def make(sl=5000, nl=12, mt='4x3', fs=500):
        m = np.zeros((nl, sl), dtype=np.float32); c25, c5 = int(2.5*fs), int(5.0*fs)
        if mt=='4x3':
            for l,c in [(0,0),(1,0),(2,0),(3,1),(4,1),(5,1),(6,2),(7,2),(8,2),(9,3),(10,3),(11,3)]: m[l,c*c25:(c+1)*c25]=1.0
        elif mt=='4x3_rhythm':
            for l,c in [(0,0),(2,0),(3,1),(4,1),(5,1),(6,2),(7,2),(8,2),(9,3),(10,3),(11,3)]: m[l,c*c25:(c+1)*c25]=1.0
            m[1,:]=1.0
        elif mt=='6x2': m[:6,:c5]=1.0; m[6:,c5:2*c5]=1.0
        elif mt=='6x2_rhythm': m[:6,:c5]=1.0; m[6:,c5:2*c5]=1.0; m[1,:]=1.0
        elif mt=='6x1': m[:6,:c5]=1.0
        elif mt=='3x1': m[:3,:c5]=1.0
        elif mt=='12x1': m[:]=1.0
        elif 'random' in mt:
            for i in range(nl): ln=int(random.uniform(0.15,0.35 if '25' in mt else 0.65)*sl); st=random.randint(0,sl-ln); m[i,st:st+ln]=1.0
        return m
    return lambda sig: (lambda mt: (make(), sig*make()))[random.choices(types, weights=probs, k=1)[0]][0], types

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, data, cfg, aug=True):
        self.d, self.cfg, self.aug = data, cfg, aug
        self.mk, self.mt = get_mask(cfg), cfg['training']['masks']
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        s = self.d[i].copy()
        if self.aug:
            if random.random()>0.5: s*=random.uniform(0.85,1.15)
            if random.random()>0.5: s+=np.random.uniform(-0.1,0.1,(12,1))
        m = self.mk(s)
        return {'partial': torch.from_numpy(s*m).float(), 'mask': torch.from_numpy(m).float(), 'target': torch.from_numpy(s).float()}

# Loss components
def pcc_loss(p, t):
    pm, tm = p.mean(-1, True), t.mean(-1, True); pc, tc = p-pm, t-tm
    return (1 - (pc*tc).sum(-1)/((pc**2).sum(-1).sqrt()*(tc**2).sum(-1).sqrt()+1e-8).clamp(-1,1)).mean()

def einthoven(p):
    I,II,III,avR,avL,avF = p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5]
    return ((I-(II-III))**2 + (avR+(I+II)/2)**2 + (avL-(I-III)/2)**2 + (avF-(II+III)/2)**2).mean()/4

def train(cfg):
    set_seed(cfg['training']['seed']); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")
    tr_d, vl_d = prepare_data(cfg)
    tr_ds, vl_ds = ECGDataset(tr_d, cfg), ECGDataset(vl_d, cfg, False)
    tr_l = torch.utils.data.DataLoader(tr_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    vl_l = torch.utils.data.DataLoader(vl_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model = ECGRecoverUNetV2(n_leads=cfg['model']['n_leads'], base_channels=cfg['model']['base_channels']).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-7)
    scaler = GradScaler()
    best_pcc, no_imp = -1.0, 0
    print("="*60)
    
    for ep in range(1, cfg['training']['epochs']+1):
        t0, tl, nt = time.time(), 0.0, 0
        model.train()
        for b in tqdm(tr_l, desc=f"Ep {ep}", leave=False, ncols=70):
            p, m, t = b['partial'].to(device, True), b['mask'].to(device, True), b['target'].to(device, True)
            opt.zero_grad()
            with autocast():
                out = model(p, m)
                lw, mw, pw, fw, ew = cfg['loss'].values()
                loss = mw*((out-t)**2).mean() + pw*pcc_loss(out,t) + fw*F.mse_loss(torch.fft.rfft(out,-1).abs(), torch.fft.rfft(t,-1).abs()) + ew*einthoven(out)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nt += 1
        
        model.eval(); vloss, vpcc, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl_l:
                p, m, t = b['partial'].to(device, True), b['mask'].to(device, True), b['target'].to(device, True)
                out = model(p, m); vloss += ((out-t)**2).mean().item(); vpcc += (1-pcc_loss(out,t)).item(); nv += 1
        tl/=nt; vloss/=nv; vpcc/=nv
        print(f"Ep {ep}/{cfg['training']['epochs']} | Train: {tl:.4f} | Val: {vloss:.4f} | PCC: {vpcc:.4f} | LR: {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")
        if vpcc > best_pcc:
            best_pcc, no_imp = vpcc, 0
            torch.save({'epoch':ep, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':opt.state_dict(), 'val_pcc':best_pcc, 'config':cfg}, Path(cfg['paths']['weights_dir'])/'ecgrecover_best.pt')
            print(f"Best saved (PCC={best_pcc:.4f})")
        else: no_imp += 1
        if no_imp >= cfg['training']['patience']: print("Early Stopping"); break
    print(f"\nBest Val PCC: {best_pcc:.4f}")

if __name__=='__main__':
    train(load_cfg('config.yaml'))