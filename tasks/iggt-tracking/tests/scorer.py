"""IGGT 3D tracking metrics (T-mIoU, T-SR, J, F, J&F), as used for the benchmark's paper table.

Predicted tracks are matched one-to-one to GT tracks by temporal IoU (greedy); unmatched GT tracks
score 0; unmatched predictions are not penalized.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def _tracks(idm, ids):
    return {i: (idm == i) for i in ids}


def _temporal_iou(p, g):
    inter = (p & g).sum()
    union = (p | g).sum()
    return inter / union if union > 0 else 0.0


def _match(preds, gts):
    pairs = [(gid, pid, _temporal_iou(gv, pv)) for gid, gv in gts.items() for pid, pv in preds.items()]
    pairs.sort(key=lambda x: -x[2])
    usedp, usedg, m = set(), set(), {}
    for gid, pid, io in pairs:
        if gid in usedg or pid in usedp or io <= 0:
            continue
        m[gid] = (pid, io)
        usedg.add(gid)
        usedp.add(pid)
    for gid in gts:
        m.setdefault(gid, (None, 0.0))
    return m


def _boundary(m, tol):
    if m.sum() == 0:
        return np.zeros_like(m)
    b = m ^ binary_erosion(m, iterations=1, border_value=0)
    return binary_dilation(b, iterations=tol)


def _f_measure(p, g, tol):
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    pb, gb = _boundary(p, tol), _boundary(g, tol)
    prec = (pb & _boundary(g, tol)).sum() / max(pb.sum(), 1)
    rec = (gb & _boundary(p, tol)).sum() / max(gb.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def score_clip(idm, inst):
    S, H, W = inst.shape
    gts = _tracks(inst, [int(i) for i in np.unique(inst) if i >= 1])
    preds = _tracks(idm, [i for i in np.unique(idm) if i >= 1])
    if not gts:
        return None
    m = _match(preds, gts)
    tol = max(1, int(round(0.008 * np.sqrt(H * H + W * W))))
    tmiou, tsr, Js, Fs = [], 0, [], []
    for gid, (pid, _) in m.items():
        gv = gts[gid]
        pv = preds[pid] if pid is not None else np.zeros_like(gv)
        tmiou.append(_temporal_iou(pv, gv))
        fj, ff, ok = [], [], True
        for f in range(S):
            if gv[f].sum() == 0:
                continue
            inter = (pv[f] & gv[f]).sum()
            union = (pv[f] | gv[f]).sum()
            j = inter / union if union > 0 else 0.0
            fj.append(j)
            ff.append(_f_measure(pv[f], gv[f], tol))
            if j <= 0.5:
                ok = False
        if fj:
            Js.append(np.mean(fj))
            Fs.append(np.mean(ff))
            tsr += int(ok)
    n = len(m)
    return dict(t_miou=float(np.mean(tmiou)), t_sr=tsr / max(n, 1),
                J=float(np.mean(Js)) if Js else 0.0, F=float(np.mean(Fs)) if Fs else 0.0,
                JF=float((np.mean(Js) + np.mean(Fs)) / 2) if Js else 0.0)
