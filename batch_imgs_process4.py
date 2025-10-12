# -*- coding: utf-8 -*-
"""
SPR 对齐+信号提取 (v4.1)
- Stage A：稀疏黑点掩膜 → 局部爬山 → 仅保存位移CSV与公共裁剪框（可选落盘对齐图）
- Stage B：按位移在线平移→公共裁剪→(R-G) 残差 → 单CSV断点续跑 + 背景ROI
- ★ 掩膜来源：允许掩膜为“整幅未对齐、未公共裁剪”的参考图坐标；会自动按公共裁剪框裁切
"""

import os
import sys
import json
import csv
import cv2
import datetime
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn, track

# ============================ 用户配置 ============================
INPUT_DIR = Path(r"D:\20251012\20251012-fixedA375-30ugPaint_JL_2")
MASK_FILE_PATH = Path(r"D:\20251012\JL2_mask.npy")
SAVE_DIR = Path(r"D:\20251012\excel")

# 可选：若希望输出对齐+裁剪后的图像用于核验，打开此开关
WRITE_ALIGNED = False
OUTPUT_SUFFIX = "_aligned_opt_v4"

# 对齐阶段参数（与 V3 一致或微调）
DEBUG_SAMPLES = 7                 # 输出若干 debug 可视化
DOWNSCALE = 2                     # 2 或 4（锚点检测/匹配分辨率）
CLAHE_CLIP = 2.0
MIN_AREA, MAX_AREA = 200, 700     # 黑点面积范围（像素）
BIN_THR_K = 1.2                   # 二值阈值 = 均值 + k*std
KERNEL_BH = 25                    # black-hat 核（奇数）
GAUSS_SMALL, GAUSS_LARGE = 3, 9   # DoG 两核（奇数）

# 局部优化参数
SEARCH_R = 12                     # small 尺度像素半径
INIT_WINDOW = 6                   # 初始步长（small 尺度）
MIN_STEP = 1
MAX_ITERS = 40
LAMBDA_TEMP = 0.01                # 时间正则 λ

# 信号提取阶段
MAX_WORKERS = max(2, mp.cpu_count()//2)
FORCE_REALIGN = False             # True 则重算位移；False 复用已有 alignment_shifts.csv
FORCE_RECOMPUTE_SIGNAL = False    # True 则无视进度CSV，强制重算全部帧
ENABLE_BACKGROUND = True          # 是否选择背景ROI并写入 mask_id=-1
PROGRESS_CSV_NAME = "progress_long_v4.csv"
BG_JSON_NAME = "background_roi.json"
# =================================================================

EXTS = (".tif",".tiff",".bmp",".png",".jpg",".jpeg",".TIF",".TIFF",".PNG",".JPG",".JPEG")

def mac_notify(title: str, subtitle: str, message: str):
    """
    在 macOS 上弹系统通知；其他平台自动忽略。
    """
    try:
        # 仅在 Darwin(macOS) 执行
        if sys.platform != "darwin":
            return
        # 安全转义双引号，避免 AppleScript 语法问题
        def esc(s: str) -> str:
            return s.replace('"', r'\"')
        script = f'display notification "{esc(message)}" with title "{esc(title)}" subtitle "{esc(subtitle)}"'
        subprocess.run(["osascript", "-e", script], check=False)
    except Exception:
        # 安静失败，不影响主流程
        pass

def list_images_sorted(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix in EXTS and not p.name.startswith('.')]
    def key(p):
        try: return (0,int(p.stem))
        except: return (1,p.stem)
    return sorted(files,key=key)

def to_gray_f32(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

def build_anchor_response(gray_f32: np.ndarray) -> np.ndarray:
    g8 = np.clip(gray_f32, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8))
    g_eq = clahe.apply(g8).astype(np.float32)
    k = max(3, KERNEL_BH | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(g_eq, cv2.MORPH_BLACKHAT, kernel)
    s = max(3, GAUSS_SMALL | 1)
    L = max(5, GAUSS_LARGE | 1)
    dog = cv2.GaussianBlur(g_eq, (s, s), 0) - cv2.GaussianBlur(g_eq, (L, L), 0)
    resp = bh + dog
    resp -= resp.min()
    if resp.max() > 1e-6: resp /= resp.max()
    return resp

def binarize_keep_area(resp: np.ndarray, min_area: int, max_area: int, k_thr: float) -> np.ndarray:
    m, sd = float(np.mean(resp)), float(np.std(resp))
    thr = m + k_thr * sd
    mask = (resp >= thr).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[labels == lab] = 255
    return keep

def downscale(img: np.ndarray, s: int) -> np.ndarray:
    if s == 1: return img
    h,w = img.shape[:2]
    return cv2.resize(img, (w//s, h//s), interpolation=cv2.INTER_AREA)

def warp_by_integer_shift(img: np.ndarray, dx: int, dy: int, border_val=0) -> np.ndarray:
    H, W = img.shape[:2]
    M = np.array([[1,0,dx],[0,1,dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W,H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def worker_build_mask(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    g = to_gray_f32(bgr)
    resp = build_anchor_response(g)
    mask = binarize_keep_area(resp, MIN_AREA, MAX_AREA, BIN_THR_K)
    return path.name, resp, mask

def overlap_score(ref_mask_small: np.ndarray, cur_mask_small: np.ndarray, dx: int, dy: int) -> int:
    H, W = ref_mask_small.shape
    x0 = max(0,  dx); x1 = min(W, W + dx)
    y0 = max(0,  dy); y1 = min(H, H + dy)
    if x1 <= x0 or y1 <= y0: return 0
    ref_roi = ref_mask_small[y0:y1, x0:x1]
    cur_roi = cur_mask_small[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return int(np.count_nonzero((ref_roi > 0) & (cur_roi > 0)))

def local_optimize_shift(ref_mask_small: np.ndarray,
                         cur_mask_small: np.ndarray,
                         start_dx: int, start_dy: int,
                         prev_dx: int, prev_dy: int,
                         search_r: int = SEARCH_R,
                         init_step: int = INIT_WINDOW,
                         min_step: int = MIN_STEP,
                         max_iters: int = MAX_ITERS,
                         lambda_temp: float = LAMBDA_TEMP) -> Tuple[int,int,int]:
    def clamp(d): return max(-search_r, min(search_r, d))
    dx, dy = clamp(start_dx), clamp(start_dy)
    best_overlap = overlap_score(ref_mask_small, cur_mask_small, dx, dy)
    best_cost = -best_overlap + lambda_temp * ((dx - prev_dx)**2 + (dy - prev_dy)**2)
    step = max(min_step, init_step)
    it = 0
    while it < max_iters and step >= min_step:
        it += 1
        improved = False
        candidates = [( step, 0), (-step, 0), (0, step), (0,-step),
                      ( step, step), ( step,-step), (-step, step), (-step,-step)]
        for ddx, ddy in candidates:
            nx, ny = clamp(dx + ddx), clamp(dy + ddy)
            ov = overlap_score(ref_mask_small, cur_mask_small, nx, ny)
            cost = -ov + lambda_temp * ((nx - prev_dx)**2 + (ny - prev_dy)**2)
            if cost < best_cost:
                dx, dy, best_cost, best_overlap = nx, ny, cost, ov
                improved = True
        if not improved:
            step //= 2
    return dx, dy, best_overlap

def common_crop_rect(shifts_xy: List[Tuple[int,int]], W: int, H: int) -> Tuple[int,int,int,int]:
    dxs = [dx for dx,_ in shifts_xy]
    dys = [dy for _,dy in shifts_xy]
    dx_min, dx_max = min(dxs), max(dxs)
    dy_min, dy_max = min(dys), max(dys)
    x0 = max(0,  dx_max)
    y0 = max(0,  dy_max)
    x1 = min(W, W + dx_min)
    y1 = min(H, H + dy_min)
    if x1 <= x0 or y1 <= y0: return 0,0,W,H
    return x0,y0,x1,y1

def compute_or_load_alignment(input_dir: Path,
                              force_realign: bool = False,
                              write_aligned: bool = False,
                              output_suffix: str = "_aligned_opt_v4"
                              ) -> Dict:
    imgs = list_images_sorted(input_dir)
    if not imgs:
        raise RuntimeError("No images found.")
    out_dir = input_dir.parent / (input_dir.name + output_suffix)
    dbg_dir = out_dir / "debug_opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)
    shifts_csv = out_dir / "alignment_shifts.csv"
    meta_json  = out_dir / "alignment_meta.json"

    if shifts_csv.exists() and meta_json.exists() and (not force_realign):
        with open(meta_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        shifts = []
        with open(shifts_csv, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                shifts.append((int(row['dx']), int(row['dy'])))
        meta['shifts'] = shifts
        return meta

    ref_bgr = cv2.imread(str(imgs[0]), cv2.IMREAD_COLOR)
    H, W = ref_bgr.shape[:2]

    print("[Stage A / Step 1] Building sparse masks in parallel ...")
    with mp.Pool(processes=MAX_WORKERS) as pool:
        results = list(pool.imap(worker_build_mask, imgs, chunksize=8))

    names = [r[0] for r in results]
    resps = [r[1] for r in results]
    masks = [r[2] for r in results]
    resps_s = [downscale(r, DOWNSCALE) for r in resps]
    masks_s = [downscale(m, DOWNSCALE) for m in masks]

    print("[Stage A / Step 2] Temporal local optimization (frame-wise) ...")
    shifts = [(0,0)]
    prev_dx = 0; prev_dy = 0
    start_dx = 0; start_dy = 0
    ns = max(5, min(9, int(DEBUG_SAMPLES)))
    sample_ids = np.linspace(0, len(imgs)-1, ns, dtype=int).tolist()

    with Progress(TextColumn("[cyan]Aligning[/]"),
                  BarColumn(), MofNCompleteColumn(),
                  TextColumn("•"), TimeRemainingColumn()) as prog:
        task = prog.add_task("", total=len(imgs)-1)
        for i in range(1, len(imgs)):
            ref_mask_s = masks_s[0]
            cur_mask_s = masks_s[i]
            dx_s, dy_s, _ = local_optimize_shift(ref_mask_s, cur_mask_s,
                                                 start_dx, start_dy, prev_dx, prev_dy,
                                                 SEARCH_R, INIT_WINDOW, MIN_STEP, MAX_ITERS, LAMBDA_TEMP)
            dx = int(dx_s * DOWNSCALE)
            dy = int(dy_s * DOWNSCALE)
            shifts.append((dx, dy))
            prev_dx, prev_dy = dx_s, dy_s
            start_dx, start_dy = dx_s, dy_s

            if i in sample_ids:
                overlay = np.zeros((H, W, 3), np.uint8)
                cur_back = warp_by_integer_shift(masks[i], dx, dy)
                overlay[...,1] = masks[0]
                overlay[...,2] = cur_back
                cv2.imwrite(str(dbg_dir / f"{i:04d}_mask_overlay.png"), overlay)
                resp_vis = (resps[i] * 255.0).astype(np.uint8)
                resp_vis = cv2.applyColorMap(resp_vis, cv2.COLORMAP_MAGMA)
                cv2.imwrite(str(dbg_dir / f"{i:04d}_resp.png"), resp_vis)
            prog.advance(task)

    x0,y0,x1,y1 = common_crop_rect(shifts, W, H)
    cw, ch = x1-x0, y1-y0
    print(f"[Stage A] Shifts: dx[min,max]=({min([d for d,_ in shifts])},{max([d for d,_ in shifts])}), "
          f"dy[min,max]=({min([d for _,d in shifts])},{max([d for _,d in shifts])})")
    print(f"[Stage A] Common crop: x0={x0}, y0={y0}, w={cw}, h={ch}")

    if write_aligned:
        print("[Stage A] Writing aligned+cropped images ...")
        with Progress(TextColumn("[cyan]Writing[/]"),
                      BarColumn(), MofNCompleteColumn(),
                      TextColumn("•"), TimeRemainingColumn()) as prog:
            task = prog.add_task("", total=len(imgs))
            for (p, (dx,dy)) in zip(imgs, shifts):
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                moved = warp_by_integer_shift(bgr, dx, dy)
                cropped = moved[y0:y1, x0:x1]
                (out_dir / p.name).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / p.name), cropped)
                prog.advance(task)

    with open(out_dir / "alignment_shifts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index","filename","dx","dy"])
        for i, (nm,(dx,dy)) in enumerate(zip(names, shifts)):
            w.writerow([i, nm, dx, dy])

    meta = {
        "version": "AlignV4.1",
        "date_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "downscale": DOWNSCALE,
        "search_r": SEARCH_R,
        "init_window": INIT_WINDOW,
        "lambda_temp": LAMBDA_TEMP,
        "crop": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": cw, "h": ch},
        # ★ 记录原始参考帧尺寸，供掩膜尺寸自适配判断
        "ref_size": {"H": int(H), "W": int(W)}
    }
    with open(out_dir / "alignment_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    meta['shifts'] = shifts
    return meta

# ---------------- 掩膜加载 ----------------
def load_mask_file(mask_file: Union[str, Path]) -> Optional[np.ndarray]:
    """返回二维整型标签图（0 为背景，其它为 label）"""
    try:
        mask_file = Path(mask_file)
        if not mask_file.exists() or (not mask_file.is_file()):
            print(f"错误：掩膜文件不存在或无效: {mask_file}"); return None
        mask_data = np.load(mask_file)
        if mask_data.ndim != 2:
            print(f"错误：掩膜应为二维数组，当前维度为 {mask_data.ndim}")
            return None
        # 保留原值（整幅或裁剪均可）
        return mask_data
    except Exception as e:
        print(f"发生错误：{e}")
        return None

def labels_to_bool_dict(label_img: np.ndarray) -> Dict[int, np.ndarray]:
    masks_dict: Dict[int, np.ndarray] = {}
    for label in np.unique(label_img):
        if label == 0:
            continue
        m = (label_img == label)
        masks_dict[int(label)] = m.astype(np.bool_)
    return masks_dict

# ---------------- 背景 ROI 的持久化 ----------------
def _safe_teardown_all_windows():
    try: cv2.destroyAllWindows()
    except Exception: pass
    for _ in range(8):
        try: cv2.waitKey(1)
        except Exception: pass

def select_background_roi(reference_bgr_cropped: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    win = "选择背景ROI：拖拽->Enter确认 / Esc跳过"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.displayOverlay(win, "拖拽矩形；Enter确认；Esc跳过", 2000)
    except Exception:
        pass
    x, y, w, h = cv2.selectROI(win, reference_bgr_cropped, showCrosshair=True, fromCenter=False)
    _safe_teardown_all_windows()
    if (x, y, w, h) == (0, 0, 0, 0):
        return None
    return (int(y), int(y+h), int(x), int(x+w))

def save_bg_roi(cache_dir: Path, roi: Tuple[int,int,int,int], ref_info: Dict):
    data = {"y1": roi[0], "y2": roi[1], "x1": roi[2], "x2": roi[3], **ref_info}
    with open(cache_dir / BG_JSON_NAME, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_bg_roi(cache_dir: Path) -> Optional[Tuple[int,int,int,int]]:
    p = cache_dir / BG_JSON_NAME
    if not p.exists(): return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return (int(d['y1']), int(d['y2']), int(d['x1']), int(d['x2']))
    except Exception:
        return None

# ---------------- 单CSV 进度管理 ----------------
def progress_csv_path(cache_dir: Path) -> Path:
    return cache_dir / PROGRESS_CSV_NAME

def read_and_clean_progress(csv_path: Path, expected_mask_count: int) -> Tuple[pd.DataFrame, Set[int]]:
    if not csv_path.exists():
        cols = ["time", "filename", "mask_id", "mean", "var"]
        return pd.DataFrame(columns=cols), set()
    df = pd.read_csv(csv_path)
    need_cols = {"time", "filename", "mask_id", "mean", "var"}
    if not need_cols.issubset(df.columns):
        raise RuntimeError(f"进度 CSV 列缺失，找到列: {df.columns.tolist()}")
    counts = df.groupby("time").size()
    done_times = set(map(int, counts[counts >= expected_mask_count].index))
    partial_times = set(map(int, counts[counts < expected_mask_count].index))
    if partial_times:
        df = df[~df["time"].isin(partial_times)].copy()
        df.to_csv(csv_path, index=False)
        print(f"已清理不完整帧 {sorted(list(partial_times))}，并回写进度 CSV。")
        counts = df.groupby("time").size()
        done_times = set(map(int, counts[counts >= expected_mask_count].index))
    return df, done_times

def append_rows_to_progress(csv_path: Path, rows: List[Tuple[int, str, int, float, float]], time_idx: int):
    for i in range(len(rows)):
        rows[i] = (time_idx, rows[i][1], rows[i][2], rows[i][3], rows[i][4])
    df = pd.DataFrame(rows, columns=["time", "filename", "mask_id", "mean", "var"])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
    else:
        with open(csv_path, 'a') as f:
            df.to_csv(f, index=False, header=False)

# ---------------- 单帧：在线移位+裁剪后计算信号 ----------------
def compute_signal_for_frame_shifted(image_path: Path,
                                     dx: int, dy: int,
                                     crop: Tuple[int,int,int,int],
                                     reference_f32_cropped: np.ndarray,
                                     masks_bool: Dict[int, np.ndarray],
                                     bg_rect: Optional[Tuple[int,int,int,int]] = None
                                     ) -> List[Tuple[int, str, int, float, float]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像: {image_path}")
    moved = warp_by_integer_shift(img, dx, dy, border_val=0)
    x0,y0,x1,y1 = crop
    cur = moved[y0:y1, x0:x1].astype(np.float32)

    residual = cur - reference_f32_cropped
    _, G_res, R_res = cv2.split(residual)
    signal_image = (R_res - G_res)

    rows: List[Tuple[int, str, int, float, float]] = []
    if bg_rect is not None:
        y1b, y2b, x1b, x2b = bg_rect
        H, W = signal_image.shape
        y1b = max(0, min(H, y1b)); y2b = max(0, min(H, y2b))
        x1b = max(0, min(W, x1b)); x2b = max(0, min(W, x2b))
        if (y2b > y1b) and (x2b > x1b):
            bg_vals = signal_image[y1b:y2b, x1b:x2b].ravel()
            rows.append((0, image_path.name, -1,
                         float(np.mean(bg_vals)) if bg_vals.size else float('nan'),
                         float(np.var(bg_vals))  if bg_vals.size else float('nan')))
        else:
            rows.append((0, image_path.name, -1, float('nan'), float('nan')))

    for mid, m in masks_bool.items():
        vals = signal_image[m]
        if vals.size == 0:
            rows.append((0, image_path.name, int(mid), float('nan'), float('nan')))
        else:
            rows.append((0, image_path.name, int(mid), float(np.mean(vals)), float(np.var(vals))))
    return rows

# ---------------- Stage B 主流程（★掩膜自动裁剪适配） ----------------
def spr_signal_extraction_v4(input_dir: Path,
                             mask_file_path: Path,
                             save_dir: Path,
                             align_meta: Dict,
                             max_workers: int = 8,
                             force_recompute: bool = False,
                             enable_background: bool = True) -> Dict[str, pd.DataFrame]:
    out_dir = Path(align_meta["output_dir"])
    cache_dir = save_dir / '_cache' / (out_dir.name + '_' + mask_file_path.stem)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = progress_csv_path(cache_dir)

    image_list = list_images_sorted(input_dir)

    shifts: List[Tuple[int,int]] = align_meta["shifts"]
    crop = align_meta["crop"]
    x0,y0,x1,y1 = crop["x0"], crop["y0"], crop["x1"], crop["y1"]
    ch, cw = (y1 - y0), (x1 - x0)

    # 参考图（裁剪后）
    ref_bgr = cv2.imread(str(image_list[0]), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise FileNotFoundError(f"参考图读取失败: {image_list[0]}")
    H_ref, W_ref = ref_bgr.shape[:2]
    ref_moved = warp_by_integer_shift(ref_bgr, shifts[0][0], shifts[0][1], border_val=0)
    ref_cropped = ref_moved[y0:y1, x0:x1]
    reference_f32_cropped = ref_cropped.astype(np.float32)

    # ★ 读取掩膜：既支持“整幅原始尺寸(H_ref,W_ref)”也支持“已裁剪尺寸(ch,cw)”
    label_img = load_mask_file(mask_file_path)
    if label_img is None:
        raise RuntimeError("掩膜加载失败。")

    if label_img.shape == (H_ref, W_ref):
        # 掩膜与原始参考同尺寸 → 直接裁剪到公共框
        label_cropped = label_img[y0:y1, x0:x1]
    elif label_img.shape == (ch, cw):
        # 掩膜已是公共裁剪尺寸
        label_cropped = label_img
    else:
        raise RuntimeError(
            f"掩膜尺寸 {label_img.shape} 与原始参考 {(H_ref, W_ref)} 或公共裁剪 {(ch, cw)} 都不一致。\n"
            f"请确认掩膜与参考图同一张原图，或改为公共裁剪尺寸。"
        )

    masks_bool = labels_to_bool_dict(label_cropped)

    # 背景 ROI（在裁剪后参考图坐标系）
    bg_rect: Optional[Tuple[int,int,int,int]] = None
    if enable_background:
        bg_rect = load_bg_roi(cache_dir)
        if bg_rect is None:
            bg_rect = select_background_roi(ref_cropped)
            if bg_rect is not None:
                save_bg_roi(cache_dir, bg_rect, {"reference": str(image_list[0].name), "crop": crop})
                print(f"背景 ROI 已保存: {bg_rect}")
            else:
                print("未选择背景 ROI，继续仅计算掩膜。")

    expected_per_frame = len(masks_bool) + (1 if (enable_background and (bg_rect is not None)) else 0)

    _, done_times = read_and_clean_progress(csv_path, expected_mask_count=expected_per_frame)

    tasks: List[Tuple[int, Path, int, int]] = []
    for t, (p, (dx,dy)) in enumerate(zip(image_list, shifts)):
        if (t not in done_times) or force_recompute:
            tasks.append((t, p, dx, dy))

    if not tasks:
        print("检测到所有帧已在进度 CSV 中完整存在，直接汇总。")
    else:
        print(f"需要处理 {len(tasks)} / {len(image_list)} 帧，线程数: {max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for (t, p, dx, dy) in tasks:
                fut = ex.submit(compute_signal_for_frame_shifted, p, dx, dy,
                                (x0,y0,x1,y1),
                                reference_f32_cropped,
                                masks_bool, bg_rect)
                futs[fut] = t
            for fut in track(as_completed(futs), total=len(futs), description="并行计算帧"):
                t = futs[fut]
                try:
                    rows = fut.result()
                    append_rows_to_progress(csv_path, rows, time_idx=t)
                except Exception as e:
                    print(f"帧 {t} 计算失败：{e}")

    if not csv_path.exists():
        raise RuntimeError("没有可用的进度 CSV，无法汇总结果。")
    long_df = pd.read_csv(csv_path)

    mean_df = long_df.pivot(index="time", columns="mask_id", values="mean").sort_index()
    var_df  = long_df.pivot(index="time", columns="mask_id", values="var").sort_index()

    def _to_int_cols(df):
        new_cols = []
        for c in df.columns:
            try: new_cols.append(int(c))
            except Exception: new_cols.append(c)
        df.columns = new_cols
        return df

    mean_df = _to_int_cols(mean_df)
    var_df  = _to_int_cols(var_df)

    if -1 in mean_df.columns:
        mean_df = mean_df.rename(columns={-1: 'Background'})
    if -1 in var_df.columns:
        var_df  = var_df.rename(columns={-1: 'Background'})

    def _sorted_cols(df):
        cols = list(df.columns)
        bg = [c for c in cols if str(c) == 'Background']
        nums = [c for c in cols if c != 'Background']
        try: nums_sorted = sorted(nums)
        except Exception: nums_sorted = nums
        return nums_sorted + bg

    mean_df = mean_df[_sorted_cols(mean_df)]
    var_df  = var_df[_sorted_cols(var_df)]

    var_df_renamed = var_df.copy()
    var_df_renamed.columns = [f"{c}_var" if c != 'Background' else 'Background_var' for c in var_df_renamed.columns]
    combined_df = pd.concat([mean_df, var_df_renamed], axis=1)

    return {"mean": mean_df, "var": var_df, "combined": combined_df,
            "cache_dir": cache_dir, "progress_csv": csv_path}

# ---------------- 保存 Excel（包含 Meta） ----------------
def save_results_to_excel_v4(result: Dict[str, pd.DataFrame],
                             save_dir: Union[str, Path],
                             mask_file_path: Union[str, Path],
                             align_meta: Dict):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    mask_file_path = Path(mask_file_path)
    out_dir = Path(align_meta["output_dir"])

    excel_name = out_dir.name + f"_{mask_file_path.stem}.xlsx"
    excel_path = save_dir / excel_name

    cache_dir = result["cache_dir"]
    bg_json = cache_dir / BG_JSON_NAME
    bg_info = {"bg_y1": None, "bg_y2": None, "bg_x1": None, "bg_x2": None}
    if bg_json.exists():
        try:
            with open(bg_json, 'r', encoding='utf-8') as f:
                d = json.load(f)
            bg_info = {"bg_y1": d.get("y1"), "bg_y2": d.get("y2"),
                       "bg_x1": d.get("x1"), "bg_x2": d.get("x2")}
        except Exception:
            pass

    meta_info = {
        'version': 'SignalExtraction4.1',
        'date_time': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'input_dir': align_meta["input_dir"],
        'alignment_output_dir': align_meta["output_dir"],
        'progress_csv': str(result["progress_csv"]),
        'crop_x0': align_meta["crop"]["x0"], 'crop_y0': align_meta["crop"]["y0"],
        'crop_x1': align_meta["crop"]["x1"], 'crop_y1': align_meta["crop"]["y1"],
        'downscale': align_meta["downscale"], 'search_r': align_meta["search_r"],
        'ref_H': align_meta["ref_size"]["H"], 'ref_W': align_meta["ref_size"]["W"],
        **bg_info,
    }
    meta_df = pd.DataFrame(meta_info, index=[0])

    with pd.ExcelWriter(excel_path) as writer:
        result["mean"].to_excel(writer, sheet_name='Mean Signal')
        result["var"].to_excel(writer, sheet_name='Variance Signal')
        result["combined"].to_excel(writer, sheet_name='Combined Signal')
        meta_df.to_excel(writer, sheet_name='Meta Info')

    print('results saved to:', str(excel_path))

# ---------------- 脚本入口 ----------------
if __name__ == '__main__':
    align_meta = compute_or_load_alignment(INPUT_DIR,
                                           force_realign=FORCE_REALIGN,
                                           write_aligned=WRITE_ALIGNED,
                                           output_suffix=OUTPUT_SUFFIX)
    result = spr_signal_extraction_v4(input_dir=INPUT_DIR,
                                      mask_file_path=MASK_FILE_PATH,
                                      save_dir=SAVE_DIR,
                                      align_meta=align_meta,
                                      max_workers=MAX_WORKERS,
                                      force_recompute=FORCE_RECOMPUTE_SIGNAL,
                                      enable_background=ENABLE_BACKGROUND)
    save_results_to_excel_v4(result, SAVE_DIR, MASK_FILE_PATH, align_meta)
        # --- 结束通知（macOS） ---
    algo_name = "细胞SPR 对齐+信号提取V4.1"
    folder_name = INPUT_DIR.name
    mac_notify(
        title=algo_name,
        subtitle=folder_name,
        message="计算完成 ✅（结果已写入 Excel 与进度CSV）"
    )
