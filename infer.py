import os, sys, yaml, argparse, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from model.unet import ECGRecoverUNetV2

import plotly.graph_objects as go
from plotly.subplots import make_subplots

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def clear_dir(p):
    p = Path(p)
    if p.exists():
        for f in p.glob("*"):
            f.is_file() and f.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)


def find_ckpt(d):
    p = Path(d)
    if not p.exists():
        return None
    best = list(p.glob("*best*.pt"))
    return best[0] if best else next(p.glob("*.pt"), None)


def bandpass(sig, fs=500.0, lo=0.5, hi=40.0, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    if len(sig) > 18:
        return filtfilt(b, a, sig).astype(np.float32)
    return sig.astype(np.float32)


def preprocess(df, cfg):
    leads    = cfg["processing"]["standard_leads"]
    L        = cfg["processing"]["target_length"]   
    fs       = cfg["processing"]["target_fs"]        
    N        = len(df)                               

    cols = [c.strip() for c in df.columns]
    skip_first = cols[0] in ["#", "Time", "time", "t", ""]
    data_cols  = df.columns[1:] if skip_first else df.columns

    norm  = np.zeros((12, L), dtype=np.float32)
    mask  = np.zeros((12, L), dtype=np.float32)
    stats = []

    for i, lead in enumerate(leads):
        col = next(
            (c for c in data_cols
             if c.strip().upper().replace(" ", "") == lead.upper()),
            None
        )
        if col is None:
            stats.append((0.0, 1.0))
            continue

        raw = pd.to_numeric(df[col], errors="coerce").values  # (N,)
        valid = ~np.isnan(raw)

        if valid.sum() < 20:
            stats.append((0.0, 1.0))
            continue

        first_row = int(np.argmax(valid))
        last_row  = int(len(valid) - np.argmax(valid[::-1]) - 1)

        segment = raw[first_row : last_row + 1].copy()

        segment = (
            pd.Series(segment)
            .interpolate("linear", limit_direction="both")
            .values.astype(np.float64)
        )

        t_first = int(round(first_row / N * L))
        t_last  = int(round((last_row + 1) / N * L))
        t_n     = t_last - t_first

        if t_n <= 0:
            stats.append((0.0, 1.0))
            continue

        seg = resample(segment, t_n).astype(np.float32)

        seg = bandpass(seg, fs=fs)

        m_raw = float(seg.mean())
        s_raw = float(seg.std()) + 1e-8
        seg   = np.clip(seg, m_raw - 5 * s_raw, m_raw + 5 * s_raw)

        # ── Z-score
        m   = float(seg.mean())
        std = float(seg.std()) + 1e-8

        norm[i, t_first:t_last] = (seg - m) / std
        mask[i, t_first:t_last] = 1.0
        stats.append((m, std))

    return norm, mask, stats


def run_inf(partial, mask, ckpt, dev):
    m = ECGRecoverUNetV2(n_leads=12, base_channels=64)
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    m.load_state_dict(state["model_state_dict"])
    m.to(dev).eval()

    with torch.no_grad():
        out = m(
            torch.from_numpy(partial).float().unsqueeze(0).to(dev),
            torch.from_numpy(mask).float().unsqueeze(0).to(dev),
        ).squeeze(0).cpu().numpy()
    return out


def save_res(out, stats, leads, p, fs):
    res = np.array([out[i] * stats[i][1] + stats[i][0] for i in range(12)])
    t   = np.arange(out.shape[1]) / fs
    pd.DataFrame(
        {"#": t, **{l: res[i] for i, l in enumerate(leads)}}
    ).to_csv(p, index=False, float_format="%.6f")


def save_plot(out, mask, stats, leads, p, fs):
    res = np.array([out[i] * stats[i][1] + stats[i][0] for i in range(12)])
    t   = np.arange(out.shape[1]) / fs

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=leads,
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    for i, lead_name in enumerate(leads):
        row = i // 3 + 1
        col = i % 3 + 1

        known     = mask[i] == 1
        recovered = mask[i] == 0

        y_known = res[i].copy()
        y_known[recovered] = np.nan
        fig.add_trace(
            go.Scatter(x=t, y=y_known, mode="lines",
                       line=dict(color="#0066cc", width=1.3),
                       name="Known", showlegend=(i == 0)),
            row=row, col=col,
        )

        y_rec = res[i].copy()
        y_rec[known] = np.nan
        fig.add_trace(
            go.Scatter(x=t, y=y_rec, mode="lines",
                       line=dict(color="#e87722", width=1.3),
                       name="Reconstructed", showlegend=(i == 0)),
            row=row, col=col,
        )

        if row == 4:
            fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="Amplitude", row=row, col=col)

    fig.update_layout(
        height=1100, width=1450,
        title_text="ECGRecover — Reconstructed 12-Lead ECG<br>"
                   "<sup>Blue = known digitized segments | Orange = model reconstruction</sup>",
        title_x=0.5, title_font_size=18,
        template="plotly_white",
        margin=dict(l=60, r=30, t=100, b=60),
    )

    html_path = Path(p).with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  → Plot saved: {html_path.name}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.yaml")
    pa.add_argument("--device", default=None)
    args = pa.parse_args()

    cfg = load_config(args.config)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    clear_dir(cfg["paths"]["output_dir"])

    ckpt = find_ckpt(cfg["paths"]["weights_dir"])
    if not ckpt:
        raise FileNotFoundError("No checkpoint in weights_dir")
    print(f"Checkpoint: {ckpt.name}")

    in_p  = Path(cfg["paths"]["input_dir"])
    out_p = Path(cfg["paths"]["output_dir"])
    files = list(in_p.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV in input_dir")
    print(f"Processing {len(files)} file(s)...\n")

    for f in files:
        print(f"{f.name}")
        try:
            df = pd.read_csv(f)
            partial, msk, stats = preprocess(df, cfg)

            for j, ld in enumerate(cfg["processing"]["standard_leads"]):
                pct = 100 * msk[j].mean()
                print(f"   {ld}: {pct:.0f}% known")

            out = run_inf(partial, msk, ckpt, dev)

            if cfg["output"]["save_csv"]:
                save_res(out, stats, cfg["processing"]["standard_leads"],
                         out_p / f"{f.stem}.csv",
                         cfg["processing"]["target_fs"])

            if cfg["output"]["save_plots"]:
                save_plot(
                    out,
                    msk,
                    stats,
                    cfg["processing"]["standard_leads"],
                    out_p / f"{f.stem}.png",
                    cfg["processing"]["target_fs"]
                )

            print(f"Done\n")

        except Exception as e:
            import traceback
            print(f"{e}")
            traceback.print_exc()

    print("All done.")


if __name__ == "__main__":
    main()