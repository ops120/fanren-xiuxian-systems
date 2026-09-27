# -*- coding: utf-8 -*-
"""extract.py — production extraction for the full roll (1179.68s → 1704.16s).

Phase A: parallel frame grab (ffmpeg double-seek) + sequential dy chain
         (grayscale template matching, anchored at t=1200.00 where geometry
         was validated: PHASE=6.0, PITCH=30.8, dy=0).
Phase B: parallel per-sample det + batch rec (ProcessPoolExecutor).
Phase C: merge, (col,g) dedup, tier assignment by g, CSV outputs.

Usage:
  python extract.py smoke --video path/to/XX.mp4   # 9 samples end-to-end (~3 min)
  python extract.py full  --video path/to/XX.mp4   # full run (~25 min, 6 workers)
"""
import os, sys, json, csv, subprocess, time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

VID = os.environ.get('FR_VIDEO', r'G:\Downloads\202609\哔哩哔哩视频\慕兰之战17.mp4')   # input video (--video overrides)
_SRC = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(_SRC)                                  # repo root (derived)
WORK = os.environ.get('FR_WORK', os.path.join(os.environ.get('TEMP', r'C:\Temp'), 'fanren_probe', 'full'))  # scratch dir (--work overrides)
OUT = os.environ.get('FR_OUT', os.path.join(PROJ, 'output'))

H, W = 1080, 1920
def _env_f(name, default):
    try: return float(os.environ[name])
    except (KeyError, ValueError): return float(default)
PITCH = _env_f('FR_PITCH', 30.80)
PHASE = _env_f('FR_PHASE', 6.00)
V = _env_f('FR_V', 259.6207 / 25.0)           # px per frame; anchor t=1200.00 dy=0
F_ANCHOR = int(_env_f('FR_ANCHOR', 30000))
LEFTS = json.loads(os.environ.get('FR_LEFTS', '[4, 241, 459, 639, 809, 1165, 1404, 1648]'))
COLW = int(_env_f('FR_COLW', 230))
TIER0 = os.environ.get('FR_TIER0', '落云宗太上长老')
BACK = json.loads(os.environ.get('FR_BACK', '[1196.32, 1192.64, 1188.96, 1185.28, 1181.60, 1179.68]'))
FWD = json.loads(os.environ.get('FR_FWD', 'null')) or [round(1203.68 + 3.68 * k, 2) for k in range(137)]

def S_of(f_abs): return V * (f_abs - F_ANCHOR)
def y_of(f_abs, g, dy=0.0): return PHASE + PITCH * g - S_of(f_abs) + dy
def g_at(f_abs, y, dy=0.0): return (y - PHASE + S_of(f_abs) - dy) / PITCH
def f_of(t): return int(round(t * 25))
def frame_path(f_abs): return os.path.join(WORK, 'f_%08d.jpg' % f_abs)
def entry_path(f_abs): return os.path.join(WORK, 'entries', 'e_%08d.json' % f_abs)
def hms(t): return '%02d:%02d:%05.2f' % (t // 3600, t % 3600 // 60, t % 60)

def grab(t):
    p = frame_path(f_of(t))
    if not os.path.exists(p):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', '%.3f' % (t - 2.0), '-i', VID,
                        '-ss', '2', '-frames:v', '1', '-q:v', '2', p], check=True)
    return p

def load_gray(f_abs):
    return cv2.cvtColor(cv2.imread(frame_path(f_abs)), cv2.COLOR_BGR2GRAY)

def vpg_list(gimg):
    out = []
    for ci in range(8):
        roi = gimg[:, LEFTS[ci]:LEFTS[ci] + COLW].astype(np.float32)
        hp = roi - cv2.GaussianBlur(roi, (1, 21), 0)
        out.append(np.abs(cv2.Sobel(hp, cv2.CV_32F, 0, 1, 3)).sum(axis=1) / 255.0)
    return out

def row_strip(gimg, f_abs, g, ci, dy, half=14):
    yc = int(round(y_of(f_abs, g, dy)))
    if yc - half < 0 or yc + half >= H: return None
    return gimg[yc - half:yc + half, LEFTS[ci]:LEFTS[ci] + COLW]

def phase_match(gref, f_ref, dy_ref, gnew, f_new, dy_pred, shared, win=40):
    offs, peaks = [], []
    for g in shared:
        for ci in range(8):
            tpl = row_strip(gref, f_ref, g, ci, dy_ref)
            if tpl is None: continue
            y_exp = y_of(f_new, g, dy_pred)
            y0, y1 = int(y_exp - win), int(y_exp + win)
            if y0 < 0 or y1 > H: continue
            band = gnew[y0:y1, LEFTS[ci]:LEFTS[ci] + COLW]
            if band.shape[0] < 28: continue
            r = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
            offs.append((y0 + int(np.argmax(r))) - int(round(y_exp)))
            peaks.append(float(r.max()))
    if not offs: return 0.0, 0, 0.0
    return float(np.median(offs)), len(offs), float(np.mean(peaks))

# ---------------- Phase A ----------------
def phase_a(times):
    for f in os.listdir(os.path.join(WORK, 'entries')):
        os.remove(os.path.join(WORK, 'entries', f))
    if os.path.exists(os.path.join(WORK, 'dy_map.json')):
        os.remove(os.path.join(WORK, 'dy_map.json'))
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(grab, times))
    print('A1 frames extracted: %d (%.0fs)' % (len(times), time.time() - t0), flush=True)
    anchor = F_ANCHOR
    locked = {anchor: 0.0}
    hist = [(anchor, 0.0)]
    g0 = load_gray(anchor)
    cache = {anchor: (g0, vpg_list(g0))}
    pending = sorted(f_of(t) for t in times if f_of(t) != anchor)
    t0 = time.time(); nflag = 0
    while pending:
        lo, hi = min(locked), max(locked)
        cands = []
        below = [f for f in pending if f < lo]
        above = [f for f in pending if f > hi]
        if below: cands.append((lo - max(below), max(below), lo))
        if above: cands.append((min(above) - hi, min(above), hi))
        _, f_new, f_ref = min(cands)
        pending.remove(f_new)
        if len(hist) >= 2:
            hh = sorted(hist, key=lambda h: abs(h[0] - f_new))[:6]
            fs = np.array([a for a, _ in hh]); vs = np.array([b for _, b in hh])
            sl, it = np.polyfit(fs, vs, 1)
            dy_pred = float(sl * f_new + it)
        else:
            dy_pred = locked[f_ref]
        if f_new not in cache:
            gimg = load_gray(f_new); cache[f_new] = (gimg, vpg_list(gimg))
        gref, vref = cache[f_ref]
        gnew, vnew = cache[f_new]
        dy_ref = locked[f_ref]
        s_lo = max(g_at(f_ref, 15, dy_ref), g_at(f_new, 15, dy_pred))
        s_hi = min(g_at(f_ref, 1065, dy_ref), g_at(f_new, 1065, dy_pred))
        cand = [g for g in range(int(np.ceil(s_lo)), int(np.floor(s_hi)) + 1)]
        if len(cand) > 80:
            step = len(cand) // 40
            cand = cand[::step][:40]
        scored = []
        for g in cand:
            e = 0.0
            for ci in range(8):
                yc = int(round(y_of(f_ref, g, dy_ref)))
                if 14 <= yc < H - 14:
                    e = max(e, vref[ci][yc - 14:yc + 15].max())
            scored.append((e, g))
        scored.sort(reverse=True)
        shared = [g for _, g in scored[:5]]
        med, n, pk = (0.0, 0, 0.0)
        if len(shared) >= 3:
            med, n, pk = phase_match(gref, f_ref, dy_ref, gnew, f_new, dy_pred, shared, 40)
        if n >= 8 and pk >= 0.5:
            dy = dy_pred + med
        else:
            dy = dy_pred; med = 0.0; nflag += 1
        locked[f_new] = dy
        hist.append((f_new, dy))
        print('A2 f%d dy=%+.1f (pred %+.1f med %+.1f n=%d pk=%.2f)' % (f_new, dy, dy_pred, med, n, pk), flush=True)
        if len(cache) > 6:
            keepset = {f_new, f_ref, min(locked), max(locked)}
            for k in [k for k in cache if k not in keepset and abs(k - f_new) > 250]:
                cache.pop(k, None)
    print('A2 chain done: %d locked, %d flagged (%.0fs)' % (len(locked), nflag, time.time() - t0), flush=True)
    json.dump({str(k): round(v, 2) for k, v in locked.items()},
              open(os.path.join(WORK, 'dy_map.json'), 'w'))
    return locked

# ---------------- Phase B ----------------
_OCR = None
def init_worker():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR
    _OCR = RapidOCR(use_angle_cls=False)
    _OCR(np.zeros((56, 120, 3), np.uint8))

def parse_rec(r):
    if r is None: return '', 0.0
    if isinstance(r, str): return r, 0.0
    try: return (r[0] if isinstance(r[0], str) else ''), float(r[1])
    except Exception: return '', 0.0

def worker_task(f_abs):
    outp = entry_path(f_abs)
    if os.path.exists(outp): return f_abs
    dy = float(json.load(open(os.path.join(WORK, 'dy_map.json')))[str(f_abs)])
    im = cv2.imread(frame_path(f_abs))
    boxes, _ = _OCR.text_detector(im)
    tboxes, nboxes = [], []
    for b in boxes:
        x0, y0 = b[0]; x1, y1 = b[2]
        if x0 > 1650 and y0 < 160: continue
        if (y1 - y0) >= 80 and (x1 - x0) >= 300 and y0 >= 2 and y1 <= 1078:
            tboxes.append(b)
        elif (x1 - x0) > 8:
            nboxes.append(b)
    crops, meta = [], []
    for b in nboxes:
        x0, y0 = b[0]; x1, y1 = b[2]
        h = y1 - y0
        col = int(np.argmin([abs(x0 - L) for L in LEFTS]))   # nearest column left (jitter-tolerant)
        xa = int(max(x0 - 2, LEFTS[col] - 40)); xb = int(min(x1 + 2, LEFTS[col] + COLW + 40))
        if xb - xa < 24: continue
        yc_box = (y0 + y1) / 2.0
        gf = g_at(f_abs, yc_box, dy)
        for g in {int(np.floor(gf)), int(np.ceil(gf))}:
            yc = y_of(f_abs, g, dy)
            if yc - 20 < 0 or yc + 20 > H: continue   # band must fit: partial rows caught at other samples
            if abs(yc_box - yc) > h / 2 + 5: continue  # box must plausibly sit on this row
            ya, yb = int(yc - 20), int(yc + 20)
            crop = im[ya:yb, xa:xb]
            if crop.size == 0: continue
            crops.append(cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            meta.append((col, int(g), float(yc)))
    tcrops = [cv2.resize(im[int(b[0][1]) - 2:int(b[2][1]) + 2, int(b[0][0]) - 2:int(b[2][0]) + 2],
                         None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for b in tboxes]
    names, titles = [], []
    if crops:
        res, _ = _OCR.text_recognizer(crops)
        for (col, g, yc), r in zip(meta, res):
            txt, sc = parse_rec(r)
            names.append(dict(col=col, g=g, y=round(yc, 1), name=txt, score=round(sc, 3)))
    if tcrops:
        res, _ = _OCR.text_recognizer(tcrops)
        for b, r in zip(tboxes, res):
            txt, sc = parse_rec(r)
            yc = (b[0][1] + b[2][1]) / 2.0
            g = int(round(g_at(f_abs, yc, dy)))
            titles.append(dict(g=g, x=int(b[0][0]), y=int(b[0][1]), w=int(b[2][0] - b[0][0]),
                               h=int(b[2][1] - b[0][1]), text=txt, score=round(sc, 3)))
    json.dump(dict(f=f_abs, names=names, titles=titles), open(outp, 'w', encoding='utf-8'),
              ensure_ascii=False)
    return f_abs

def phase_b(locked, workers=6):
    tasks = sorted(int(k) for k in locked)
    todo = [f for f in tasks if not os.path.exists(entry_path(f))]
    print('B: %d samples (%d to run), %d workers' % (len(tasks), len(todo), workers), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as ex:
        for i, f in enumerate(ex.map(worker_task, todo)):
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print('B %d/%d (%.0fs)' % (i + 1, len(todo), time.time() - t0), flush=True)
    print('B done (%.0fs)' % (time.time() - t0), flush=True)

# ---------------- Phase C ----------------
def phase_c(locked):
    entries, titles = [], []
    for f_abs in sorted(int(k) for k in locked):
        if not os.path.exists(entry_path(f_abs)):
            print('C: MISSING entry f%d (worker failed?)' % f_abs, flush=True); continue
        d = json.load(open(entry_path(f_abs), encoding='utf-8'))
        for n in d['names']:
            entries.append(dict(f=f_abs, sec=hms(f_abs / 25.0), **n))
        for t in d['titles']:
            titles.append(dict(f=f_abs, sec=hms(f_abs / 25.0), **t))
    # per-sample new-row counts -> tail cut (ending card guard)
    seen = set(); news = {}
    for e in sorted(entries, key=lambda e: e['f']):
        k = (e['col'], e['g'])
        news[e['f']] = news.get(e['f'], 0) + (1 if k not in seen else 0)
        seen.add(k)
    good = [f for f in sorted(news) if news[f] >= 10]
    f_cut = max(good) if good else max(news)
    dropped = sorted(f for f in news if f > f_cut)
    entries = [e for e in entries if e['f'] <= f_cut]
    titles = [t for t in titles if t['f'] <= f_cut]
    # names: dedup by (col,g) keep earliest sample
    best = {}
    for e in sorted(entries, key=lambda e: e['f']):
        k = (e['col'], e['g'])
        if k not in best: best[k] = e
    # chain-slip merge: same col, same stripped text, |dg|<=2
    def norm(s): return ''.join(s.split())
    keep = {}
    for (col, g), e in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        k2 = (col, norm(e['name']))
        dup = False
        for (c2, g2) in list(keep):
            if c2 == col and abs(g2 - g) <= 2 and norm(keep[(c2, g2)]['name']) == k2[1] and k2[1]:
                dup = True; break
        if not dup: keep[(col, g)] = e
    import re as _re
    inv = _re.compile(r'[​-‏﻿\s]')
    names = [e for e in keep.values() if inv.sub('', e['name'])]
    # titles: merge same-text within |dg|<=3 (same title seen at consecutive samples)
    tbest = []
    for t in sorted(titles, key=lambda t: t['f']):
        if not t['text'].strip(): continue
        if any(abs(u['g'] - t['g']) <= 3 and u['text'].strip() == t['text'].strip() for u in tbest):
            continue
        tbest.append(t)
    tiers = sorted(tbest, key=lambda t: t['g'])
    # tier assignment by g
    def tier_of(g):
        cur = TIER0
        for t in tiers:
            if t['g'] <= g + 2: cur = t['text'].strip() or cur
            else: break
        return cur
    for e in names: e['tier'] = tier_of(e['g'])
    names.sort(key=lambda e: (e['f'], e['y'], e['col']))
    # outputs
    p1 = os.path.join(OUT, 'names.csv')
    with open(p1, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['当前秒', '当前帧', '抬头', '用户名'])
        for e in names: w.writerow([e['sec'], e['f'], e['tier'], e['name']])
    p2 = os.path.join(OUT, 'pending_review.csv')
    with open(p2, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['当前秒', '当前帧', '抬头', '用户名', 'score', '状态'])
        for e in names:
            if e['score'] < 0.75:
                w.writerow([e['sec'], e['f'], e['tier'], e['name'], e['score'],
                            'pending' if e['score'] < 0.60 else 'low_conf'])
    p3 = os.path.join(OUT, 'tiers.csv')
    with open(p3, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['抬头', '首次可见秒', '首次可见帧', 'score'])
        for t in tiers: w.writerow([t['text'].strip(), t['sec'], t['f'], t['score']])
    st = {}
    for e in names:
        k = 'ok' if e['score'] >= 0.75 else ('low_conf' if e['score'] >= 0.60 else 'pending')
        st[k] = st.get(k, 0) + 1
    print('C: names=%d rows g%d..g%d titles=%d' % (
        len(names), min(e['g'] for e in names), max(e['g'] for e in names), len(tiers)), flush=True)
    ntot = max(len(names), 1)
    gray_n = 100.0 * (st.get('low_conf', 0) + st.get('pending', 0)) / ntot
    print('C: status=%s gray=%.1f%% tail_cut=f%d dropped=%s' % (st, gray_n, f_cut, dropped), flush=True)
    print('C: files -> %s' % p1, flush=True)
    json.dump(dict(names=names, tiers=tiers, phase=locked),
              open(os.path.join(OUT, 'full_run.json'), 'w', encoding='utf-8'), ensure_ascii=False)

def main():
    import argparse
    ap = argparse.ArgumentParser(prog='extract.py', description='凡人修仙传赞助花名册提取系统 — 视频滚动赞助名单提取')
    ap.add_argument('mode', choices=['smoke', 'full'],
                    help='smoke=9采样帧自测(约3分钟) / full=全量提取(约25分钟)')
    ap.add_argument('--video', metavar='PATH', help='输入视频路径(默认用源码顶部 VID 常量)')
    ap.add_argument('--out', metavar='DIR', help='输出目录(默认 output/)')
    ap.add_argument('--work', metavar='DIR', help='中间产物目录: 采样帧/相位表缓存(默认系统TEMP)')
    ap.add_argument('--config', metavar='JSON', help='视频配置JSON: {video,pitch,phase,v,anchor,back,fwd,tier0,lefts,colw,work,out}')
    args = ap.parse_args()
    if args.config:
        _CK = dict(video='VID', pitch='PITCH', phase='PHASE', v='V', anchor='F_ANCHOR',
                   back='BACK', fwd='FWD', tier0='TIER0', lefts='LEFTS', colw='COLW',
                   work='WORK', out='OUT')
        cfg = json.load(open(args.config, encoding='utf-8'))
        for kk, val in cfg.items():
            kk = kk.lower()
            if kk in _CK:
                os.environ['FR_' + kk.upper()] = val if isinstance(val, str) else json.dumps(val)
                globals()[_CK[kk]] = val
    global VID, WORK, OUT
    if args.video: VID = args.video
    if args.out: OUT = args.out
    if args.work:
        WORK = args.work
        os.environ['FR_WORK'] = args.work       # propagate to worker processes
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(WORK, 'entries'), exist_ok=True)
    if not os.path.exists(VID):
        print('ERROR: video not found: %s' % VID)
        sys.exit(2)
    print('mode=%s | video=%s | out=%s | work=%s' % (args.mode, VID, OUT, WORK), flush=True)
    nworkers = 6
    if args.mode == 'smoke':
        times = [1200.00] + BACK[:4] + FWD[:4]
    else:
        times = [1200.00] + BACK + FWD
    locked = phase_a(times)
    phase_b(locked, nworkers)
    phase_c(locked)
    print('ALL DONE', flush=True)

if __name__ == '__main__':
    main()
