import os
import sys
import time
import queue
import argparse
import threading
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---- multi-core env vars (must be set before numpy/BLAS import) ----
_NCPU = os.cpu_count() or 4
os.environ.setdefault('OMP_NUM_THREADS', str(_NCPU))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_NCPU))
os.environ.setdefault('MKL_NUM_THREADS', str(_NCPU))

import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, filedialog

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    from scipy.signal import find_peaks as _sp_find_peaks
except Exception:
    _sp_find_peaks = None
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    HAS_SK = True
except Exception:
    HAS_SK = False

# ============================== NOE SDK ==============================
SDK_ROOT = "/home/orangepi/ai_model_hub_25_Q3"
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)
try:
    from utils.NOE_Engine import EngineInfer
except Exception as e:
    print(f"[FATAL] Cannot import NOE SDK (SDK_ROOT={SDK_ROOT}): {e}")
    sys.exit(1)


# ============================== constants ==============================
BODY_KPT_NUM = 18
FACE_KPT_NUM = 5
BODY_DIM = 36          # 18 points * (x, y)
FACE_DIM = 10          # 5 points * (x, y)
TOTAL_DIM = BODY_DIM + FACE_DIM   # 46

POSE_NET_H, POSE_NET_W = 256, 360
POSE_STRIDE = 8
POSE_HM_H = POSE_NET_H // POSE_STRIDE   # 32
NUM_HM, NUM_PAF = 19, 38

FACE_NET = 368
FACE_NUM_PEAKS = 71
FACE_KEY_INDICES = [30, 36, 45, 51, 57]   # nose / outer eye corners / lips (adjustable)

HIP_R, HIP_L = 8, 11
KPT_SCORE_THRESH = 0.10
FACE_SCORE_THRESH = 0.05
MIN_BODY_POINTS = 6

# person selection (business rule): target = person with the LARGEST face
FACE_STICKY_IOU = 0.3    # keep previous target face if still detected (anti-flicker)

# chart 3/4 signal dims: right wrist (COCO idx4 -> flat 8,9). use 4,5 to match old plots
SIGNAL_DIM_X, SIGNAL_DIM_Y = 8, 9

# periodicity: minimum time lag in frames. a digital-human loop is never shorter
# than ~5s, so lags below this are meaningless and excluded from the search
MIN_LAG_FRAMES = 150

CHART_EVERY = 300        # refresh live charts every N frames
PREVIEW_W = 560
PREVIEW_MAX_H = 520      # max preview height, keeps bottom bar visible for portrait videos

# ---- GenD deepfake (CPU) ----
GEND_INPUT_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
FAKE_THRESH = 0.65        # frame avg > 0.5 -> fake (matches C++ reference)
SUSPECT_THRESH = 0.3
MIN_FACE_SIZE = 80       # px (C++ QualityConfig.min_face_size)
YAW_THRESH, PITCH_THRESH, ROLL_THRESH = 25.0, 20.0, 15.0   # C++ QualityConfig
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
DETECTOR_URL = ("https://huggingface.co/datasets/theanhntp/Liblib/resolve/"
                "ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/"
                "buffalo_l/det_10g.onnx")

BODY_SKELETON = [
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]


# ============================== small utilities ==============================
def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(path)}")
    return path


def _entropy(p):
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def _xywh_iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def person_roi_from_face(face_box, W, H, body_faces=9.0, width_faces=2.5):
    """Expand the target face box to a region covering that person's body."""
    x, y, w, h = face_box
    cx = x + w * 0.5
    top = max(0, y - 1.0 * h)
    bottom = min(H, y + body_faces * h)
    half_w = width_faces * w
    return (max(0.0, cx - half_w), top, min(W, cx + half_w), bottom)


def find_peaks_compat(scores, height=0.4, distance=15, prominence=0.08):
    if _sp_find_peaks is not None:
        return _sp_find_peaks(scores, height=height, distance=distance,
                              prominence=prominence)
    s = np.asarray(scores, np.float64)
    cand = [i for i in range(1, len(s) - 1)
            if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] >= height]
    kept = []
    for i in cand:
        if kept and i - kept[-1] < distance:
            if s[i] > s[kept[-1]]:
                kept[-1] = i
        else:
            kept.append(i)
    idx = np.array(kept, int)
    return idx, {'peak_heights': s[idx] if idx.size else np.array([])}


def kmeans_fit_predict(data, k, seed=42, n_init=10, max_iter=300):
    """Return (labels, inertia). numpy fallback when sklearn is missing."""
    if HAS_SK:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        labels = km.fit_predict(data)
        return labels, float(km.inertia_)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = data[rng.choice(len(data), k, replace=False)].copy()
        labels = np.zeros(len(data), int)
        for _it in range(max_iter):
            d = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = d.argmin(1)
            done = np.array_equal(new_labels, labels) and _it > 0
            labels = new_labels
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = data[m].mean(0)
            if done:
                break
        inertia = float(((data - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels, best_inertia


def silhouette_compat(data, labels):
    if HAS_SK:
        return float(silhouette_score(data, labels))
    rng = np.random.default_rng(0)
    m = min(500, len(data))
    idx = rng.choice(len(data), m, replace=False)
    X, L = data[idx], np.asarray(labels)[idx]
    uniq = np.unique(L)
    if len(uniq) < 2:
        return 0.0
    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    s = np.zeros(m)
    for i in range(m):
        same = L == L[i]
        a = D[i, same].sum() / max(int(same.sum()) - 1, 1)
        b = min(D[i, L == u].mean() for u in uniq if u != L[i])
        mx = max(a, b)
        s[i] = (b - a) / mx if mx > 0 else 0.0
    return float(s.mean())


def calinski_compat(data, labels):
    if HAS_SK:
        return float(calinski_harabasz_score(data, labels))
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2 or len(data) <= k:
        return 0.0
    mu = data.mean(0)
    within = sum(((data[labels == u] - data[labels == u].mean(0)) ** 2).sum() for u in uniq)
    between = sum(int((labels == u).sum()) * ((data[labels == u].mean(0) - mu) ** 2).sum()
                  for u in uniq)
    if within <= 0:
        return 0.0
    return float(between / within * (len(data) - k) / (k - 1))


# ============================== preprocessing ==============================
def preprocess_pose_fixed(img, net_h=POSE_NET_H, net_w=POSE_NET_W,
                          img_mean=(128, 128, 128), pad_value=(0, 0, 0),
                          img_scale=1.0 / 256):
    h, w = img.shape[:2]
    scale = min(net_h / h, net_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    pad_left = pad_top = 0
    padded = cv2.copyMakeBorder(resized, pad_top, net_h - new_h,
                                pad_left, net_w - new_w,
                                cv2.BORDER_CONSTANT, value=pad_value)
    tensor = (padded.astype(np.float32) - np.array(img_mean, np.float32)) * img_scale
    tensor = tensor.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), scale, (pad_left, pad_top)


def preprocess_face_crop(crop):
    img = cv2.resize(crop, (FACE_NET, FACE_NET))
    t = img.astype(np.float32) / 256.0 - 0.5
    return np.ascontiguousarray(t.transpose(2, 0, 1)[None], dtype=np.float32)


def reshape_output(arr, num_parts, h):
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] == num_parts:
        return a
    flat = np.asarray(arr, np.float32).reshape(-1)
    if flat.size % (num_parts * h) != 0:
        raise ValueError(f"output size {flat.size} not divisible by {num_parts}x{h}")
    return flat.reshape(num_parts, h, flat.size // (num_parts * h))


def pick_heatmap_paf(outputs, h, num_hm=NUM_HM, num_pa=NUM_PAF):
    hm = pa = None
    for o in outputs:
        a = np.asarray(o)
        c = a.shape[1] if a.ndim == 4 else (a.shape[0] if a.ndim == 3 else None)
        if c == num_hm:
            hm = reshape_output(a, num_hm, h)
        elif c == num_pa:
            pa = reshape_output(a, num_pa, h)
    if hm is not None and pa is not None:
        return hm, pa
    a2, a1 = np.asarray(outputs[-2]), np.asarray(outputs[-1])
    if a1.size == 2 * a2.size:
        hm_raw, pa_raw = a2, a1
    elif a2.size == 2 * a1.size:
        hm_raw, pa_raw = a1, a2
    else:
        hm_raw, pa_raw = a2, a1
    return reshape_output(hm_raw, num_hm, h), reshape_output(pa_raw, num_pa, h)


def pick_face_heatmaps(outputs):
    best = None
    for o in reversed(list(outputs)):
        a = np.asarray(o)
        if a.size % FACE_NUM_PEAKS == 0:
            best = a
            break
    if best is None:
        best = np.asarray(outputs[-1])
    flat = best.reshape(-1)
    h = int(round(np.sqrt(flat.size / FACE_NUM_PEAKS)))
    if h * h * FACE_NUM_PEAKS != flat.size:
        raise ValueError(f"face output size {flat.size} is not {FACE_NUM_PEAKS}xhxh")
    return flat.reshape(FACE_NUM_PEAKS, h, h)


def _soft_argmax_peak(hmap, k):
    y, x = np.unravel_index(int(np.argmax(hmap)), hmap.shape)
    score = float(hmap[y, x])
    Hm, Wm = hmap.shape
    y0, y1 = max(0, y - k), min(Hm, y + k + 1)
    x0, x1 = max(0, x - k), min(Wm, x + k + 1)
    patch = np.maximum(hmap[y0:y1, x0:x1].astype(np.float64), 0)
    s = patch.sum()
    if s > 1e-6:
        wy = (patch.sum(1) * np.arange(y0, y1)).sum() / s
        wx = (patch.sum(0) * np.arange(x0, x1)).sum() / s
    else:
        wy, wx = float(y), float(x)
    return wx, wy, score


def decode_pose_heatmaps(heatmaps, k=2):
    kpts = np.zeros((BODY_KPT_NUM, 3), np.float32)
    for i in range(BODY_KPT_NUM):
        wx, wy, score = _soft_argmax_peak(heatmaps[i], k)
        kpts[i] = (wx * POSE_STRIDE, wy * POSE_STRIDE, score)
    return kpts


def compute_peaks_from_heatmaps(heatmaps, k=1):
    peaks = np.zeros((heatmaps.shape[0], 3), np.float32)
    for i in range(heatmaps.shape[0]):
        wx, wy, score = _soft_argmax_peak(heatmaps[i], k)
        peaks[i] = (wx, wy, score)
    return peaks


def pose_to_keypoints(pose, frame_w, frame_h, scale=1.0, pad=(0.0, 0.0)):
    if pose is None:
        return np.zeros((BODY_KPT_NUM, 2), np.float32), np.zeros(BODY_KPT_NUM, bool)
    arr = np.asarray(pose, np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr[:BODY_KPT_NUM]
    xy = arr[:, :2].copy()
    sc = arr[:, 2].copy() if arr.shape[1] >= 3 else np.ones(BODY_KPT_NUM, np.float32)
    x = (xy[:, 0] - pad[0]) / max(scale, 1e-12)
    y = (xy[:, 1] - pad[1]) / max(scale, 1e-12)
    norm = np.stack([x / frame_w, y / frame_h], axis=1).astype(np.float32)
    valid = (sc >= KPT_SCORE_THRESH) & (np.abs(xy).sum(axis=1) > 0)
    norm[~valid] = 0.0
    return norm, valid


def face_to_normalized(pts_px, frame_w, frame_h):
    norm = np.zeros((FACE_KPT_NUM, 2), np.float32)
    valid = np.zeros(FACE_KPT_NUM, bool)
    if pts_px is None:
        return norm, valid
    pts_px = np.asarray(pts_px, np.float32)
    valid = pts_px[:, 2] >= FACE_SCORE_THRESH
    norm[:, 0] = pts_px[:, 0] / frame_w
    norm[:, 1] = pts_px[:, 1] / frame_h
    norm[~valid] = 0.0
    return norm, valid


def build_frame_features(body_norm, body_valid, face_norm, face_valid):
    hip_ids = [i for i in (HIP_R, HIP_L) if body_valid[i]]
    if not hip_ids or int(body_valid.sum()) < MIN_BODY_POINTS:
        return None
    center = body_norm[hip_ids].mean(0)
    feat = np.zeros(TOTAL_DIM, np.float64)
    for i in range(BODY_KPT_NUM):
        if body_valid[i]:
            feat[2 * i:2 * i + 2] = body_norm[i] - center
    for j in range(FACE_KPT_NUM):
        if face_valid[j]:
            feat[BODY_DIM + 2 * j:BODY_DIM + 2 * j + 2] = face_norm[j] - center
    return feat


# ============================== NPU models ==============================
class NoeModel:
    def __init__(self, model_path):
        require_file(model_path)
        try:
            self.engine = EngineInfer(model_path)
        except Exception as e:
            raise RuntimeError(f"EngineInfer load error: {model_path}: {e}")
        for attr in ("handle", "graph", "model", "job", "net"):
            h = getattr(self.engine, attr, None)
            if h is not None and not bool(h):
                raise RuntimeError(f"Model load failed (empty handle {attr}): {model_path}")
        self._closed = False

    def forward(self, x):
        return list(self.engine.forward([np.ascontiguousarray(x, dtype=np.float32)]))

    def close(self):
        if not self._closed:
            try:
                self.engine.clean()
            except Exception:
                pass
            self._closed = True


class PoseEstimator:
    def __init__(self, model):
        self.model = model

    def _infer_norm(self, frame):
        x, scale, pad = preprocess_pose_fixed(frame)
        outs = self.model.forward(x)
        hm_raw, pa_raw = pick_heatmap_paf(outs, POSE_HM_H)
        heatmaps = reshape_output(hm_raw, NUM_HM, POSE_HM_H)
        kpts = decode_pose_heatmaps(heatmaps)
        return pose_to_keypoints(kpts, frame.shape[1], frame.shape[0], scale, pad)

    def __call__(self, frame, roi=None):
        """roi=None: legacy full-frame path. roi=(x0,y0,x1,y1): run pose on the
        selected person's region, then map keypoints back to full-frame coords."""
        H, W = frame.shape[:2]
        if roi is None:
            return self._infer_norm(frame)
        x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
        x1, y1 = min(W, int(roi[2])), min(H, int(roi[3]))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return self._infer_norm(frame)
        body_norm, body_valid = self._infer_norm(crop)
        cw, ch = crop.shape[1], crop.shape[0]
        out = np.zeros_like(body_norm)
        out[:, 0] = body_norm[:, 0] * cw / W + x0 / W
        out[:, 1] = body_norm[:, 1] * ch / H + y0 / H
        out[~body_valid] = 0.0
        return out, body_valid


class FaceLandmarkDetector:
    def __init__(self, model, margin=0.2, detect_w=480):
        self.model = model
        self.margin = margin
        self.detect_w = detect_w
        self._prev_box = None
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")

    def __call__(self, frame):
        """Returns (pts_px, face_box, n_faces).
        Business rule: target = person with the LARGEST face."""
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downscale for faster Haar detection, scale coords back later
        ds = min(1.0, self.detect_w / W)
        if ds < 1.0:
            g = cv2.resize(gray, (int(W * ds), int(H * ds)))
        else:
            g = gray
        faces = self.cascade.detectMultiScale(g, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(40, 40))
        if faces is None or len(faces) == 0:
            return None, None, 0
        boxes = [tuple(int(v) for v in (np.array(f) / ds)) for f in faces]
        # sticky: keep the previous target if still detected (anti A/B flicker)
        box = None
        if self._prev_box is not None:
            for b in boxes:
                if _xywh_iou(self._prev_box, b) >= FACE_STICKY_IOU:
                    box = b
                    break
        if box is None:
            box = max(boxes, key=lambda b: b[2] * b[3])   # largest face
        self._prev_box = box
        x, y, w, h = box
        mx, my = int(w * self.margin), int(h * self.margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None, None, len(boxes)
        outs = self.model.forward(preprocess_face_crop(crop))
        hm = pick_face_heatmaps(outs)
        peaks = compute_peaks_from_heatmaps(hm)
        sel = peaks[FACE_KEY_INDICES]
        sx, sy = (x1 - x0) / hm.shape[2], (y1 - y0) / hm.shape[1]
        pts = np.zeros((FACE_KPT_NUM, 3), np.float32)
        pts[:, 0] = x0 + sel[:, 0] * sx
        pts[:, 1] = y0 + sel[:, 1] * sy
        pts[:, 2] = sel[:, 2]
        return pts, (x, y, w, h), len(boxes)


# ============================== GenD deepfake (CPU, ONNX Runtime) ==============================
def download_file(url, save_path):
    save_path = str(save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    print(f"[GenD] Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[GenD] Saved: {save_path}")


def clip_preprocess(image_rgb, size=GEND_INPUT_SIZE):
    """CLIP-style preprocessing (exactly detect_onnx.py preprocess_image)."""
    if HAS_PIL:
        img = Image.fromarray(image_rgb).resize((size, size), Image.BICUBIC)
        img = np.asarray(img, np.float32)
    else:
        img = cv2.resize(image_rgb, (size, size),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = img / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img, np.float32)


def _pt_dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def estimate_pose_angles(lm5):
    """Port of C++ FaceQuality::estimatePose.
    lm5: (5,2) left_eye, right_eye, nose, left_mouth, right_mouth."""
    le, re, nose, lmo, rmo = [np.asarray(p, np.float64) for p in lm5]
    eye_c = (le + re) / 2.0
    mouth_c = (lmo + rmo) / 2.0
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    eye_width = _pt_dist(le, re)
    nose_off_x = (nose[0] - eye_c[0]) / max(eye_width, 1e-6)
    yaw = float(np.degrees(np.arcsin(np.clip(nose_off_x * 2.0, -1.0, 1.0))))
    face_height = _pt_dist(eye_c, mouth_c)
    nose_off_y = (nose[1] - eye_c[1]) / max(face_height, 1e-6)
    pitch = (nose_off_y - 0.45) * 90.0
    return yaw, pitch, roll


def assess_face_quality(bbox, lm5, W, H, min_size=MIN_FACE_SIZE):
    """Port of C++ FaceQuality::assessQuality -> 'valid' or reject reason."""
    x1, y1, x2, y2 = bbox[:4]
    w, h = x2 - x1, y2 - y1
    if min(w, h) < min_size:
        return 'too_small'
    if lm5 is not None:
        yaw, pitch, roll = estimate_pose_angles(lm5)
        if abs(yaw) > YAW_THRESH:
            return 'yaw_too_large'
        if abs(pitch) > PITCH_THRESH:
            return 'pitch_too_large'
        if abs(roll) > ROLL_THRESH:
            return 'roll_too_large'
    return 'valid'


def align_face_local(img, lm5, scale=1.3):
    """5-point rotation + eye-mouth crop. Fallback when the user's
    detector.align_face is not importable. lm5: (5,2)."""
    le, re, nose, lmo, rmo = [np.asarray(p, np.float64) for p in lm5]
    eye_c = (le + re) / 2.0
    mouth_c = (lmo + rmo) / 2.0
    dy, dx = mouth_c[1] - eye_c[1], mouth_c[0] - eye_c[0]
    angle = float(np.degrees(np.arctan2(dy, dx))) - 90.0
    d = float(np.hypot(dx, dy))               # eye-to-mouth distance
    size = max(16, int(round(2.22 * d * scale)))   # face height ~= d / 0.45
    center = ((eye_c[0] + mouth_c[0]) / 2.0, (eye_c[1] + mouth_c[1]) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += size / 2.0 - center[0]
    M[1, 2] += size / 2.0 - center[1]
    aligned = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, M


def align_face_box(img, bbox, scale=1.3):
    """Square crop around bbox with margin (Haar fallback, no rotation)."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(16.0, max(x2 - x1, y2 - y1) * scale)
    H, W = img.shape[:2]
    xa, ya = int(round(cx - size / 2)), int(round(cy - size / 2))
    xb, yb = int(round(cx + size / 2)), int(round(cy + size / 2))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(W, xb), min(H, yb)
    return img[ya:yb, xa:xb]


# ============ bundled RetinaFace + align_face (fused from detect_onnx.py project) ============
# Verbatim from src/retinaface.py + detector.py::align_face. Deps: numpy + cv2 + onnxruntime.
try:
    import onnxruntime
except Exception:
    onnxruntime = None


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class RetinaFace:
    def __init__(self, model_file=None, session=None, providers=["CPUExecutionProvider"]):
        if onnxruntime is None:
            raise RuntimeError("onnxruntime not available (pip install onnxruntime)")
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        if self.session is None:
            assert self.model_file is not None
            assert os.path.exists(self.model_file)
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
            sess_options.inter_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
            self.session = onnxruntime.InferenceSession(self.model_file, sess_options)
            self.session.set_providers(providers)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get("det_thresh", None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in detection model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1],
                                          axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors,
                                              axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


def align_face(img, landmarks, target_size=None, scale=1.3, mask=None):
    """5-point landmark face alignment (verbatim from detector.py)."""
    dst = np.array(
        [
            [0.34, 0.46],
            [0.66, 0.46],
            [0.5, 0.64],
            [0.37, 0.82],
            [0.63, 0.82],
        ],
        dtype=np.float32,
    )
    if target_size is None:
        desired_dists = np.linalg.norm(landmarks[:, None, :] - landmarks[None, :, :],
                                       axis=-1)
        dst_dists = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=-1)
        upper_triangle_indices = np.triu_indices(len(dst), k=1)
        dst_dists = dst_dists[upper_triangle_indices]
        desired_dists = desired_dists[upper_triangle_indices]
        approx_size = np.round(np.mean(desired_dists / dst_dists) * scale).astype(int)
        target_size = (approx_size, approx_size)
    dst[:, 0] = dst[:, 0] * target_size[0]
    dst[:, 1] = dst[:, 1] * target_size[1]
    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
    src = landmarks.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    img = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.warpAffine(mask, M, target_size, flags=cv2.INTER_NEAREST)
    return img, mask


class GendDeepfakeDetector:
    """GenD deepfake detection, CPU (ONNX Runtime CPUExecutionProvider).
    Face backend:
      1) bundled RetinaFace + 5-point align_face (fused from detect_onnx.py)
      2) fallback: OpenCV Haar (no alignment / no pose filter, degraded)
    NPU port later: only _load_gend() and _infer_gend() need to change."""

    def __init__(self, gend_path, det_path, gend_root='.', det_thresh=0.5,
                 face_scale=1.3, min_face=MIN_FACE_SIZE, quality_filter=True,
                 log=print):
        self.log = log
        self.det_thresh = det_thresh
        self.face_scale = face_scale
        self.min_face = min_face
        self.quality_filter = quality_filter
        self.session = None
        self.input_name = None
        self.detector = None
        self.align_fn = None
        self.cascade = None
        self.backend = 'none'
        self._load_gend(gend_path)
        self._load_face_backend(det_path, gend_root)

    def _load_gend(self, gend_path):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"onnxruntime not available (pip install onnxruntime): {e}")
        require_file(gend_path)
        # CPU for now; NPU (.cix via NOE SDK) comes later
        self.session = ort.InferenceSession(gend_path,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.log(f"[GenD] ONNX loaded (CPU): {gend_path} | "
                 f"input {self.session.get_inputs()[0].name} "
                 f"{self.session.get_inputs()[0].shape}")

    def _load_face_backend(self, det_path, gend_root=None):
        # backend 1: bundled RetinaFace + 5-point align_face (fused from the
        # detect_onnx.py project; same numerics as the PC reference)
        if not det_path or not os.path.isfile(det_path):
            for cand in ('model/det_10g.onnx',
                         'weights/models/buffalo_l/det_10g.onnx',
                         'det_10g.onnx'):
                for base in ('', os.path.dirname(os.path.abspath(__file__))):
                    p = os.path.join(base, cand) if base else cand
                    if os.path.isfile(p):
                        det_path = p
                        break
                if det_path and os.path.isfile(det_path):
                    break
        if not det_path or not os.path.isfile(det_path):
            det_path = os.path.join('model', 'det_10g.onnx')
            self.log(f"[GenD] det model not found, downloading -> {det_path}")
            download_file(DETECTOR_URL, det_path)
        try:
            self.detector = RetinaFace(det_path, providers=['CPUExecutionProvider'])
            self.detector.prepare(ctx_id=0, nms_thresh=0.4,
                                  input_size=(640, 640), det_thresh=self.det_thresh)
            self.align_fn = align_face
            self.backend = 'retinaface'
            self.log(f"[GenD] face backend: RetinaFace (fused) ({det_path})")
            return
        except Exception as e:
            self.log(f"[GenD] RetinaFace backend unavailable ({e})")
        # backend 2: Haar fallback
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    'haarcascade_frontalface_default.xml')
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")
        self.backend = 'haar'
        self.log("[GenD] face backend: OpenCV Haar (no alignment, pose filter disabled)")

    def _infer_gend(self, aligned_bgr):
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        t = clip_preprocess(rgb)
        outs = self.session.run(None, {self.input_name: t})
        logits = np.asarray(outs[0], np.float64).reshape(-1)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return float(probs[1])     # P(fake) = softmax[1] (detect_onnx.py)

    def detect_frame(self, image_bgr, max_faces=None):
        """Returns list of (bbox_xyxy, p_fake, is_valid, reason)."""
        H, W = image_bgr.shape[:2]
        faces = []
        if self.backend == 'retinaface':
            xyxy, landmarks = self.detector.detect(image_bgr)
            if xyxy is None or len(xyxy) == 0:
                return faces
            order = sorted(range(len(xyxy)),
                           key=lambda i: ((xyxy[i][2] - xyxy[i][0]) *
                                          (xyxy[i][3] - xyxy[i][1])), reverse=True)
            if max_faces:
                order = order[:max_faces]
            for i in order:
                bbox = tuple(float(v) for v in xyxy[i][:4])
                lm5 = np.asarray(landmarks[i], np.float32).reshape(5, 2)
                reason = (assess_face_quality(bbox, lm5, W, H, self.min_face)
                          if self.quality_filter else 'valid')
                if reason != 'valid':
                    faces.append((bbox, 0.0, False, reason))
                    continue
                try:
                    aligned, _ = self.align_fn(image_bgr, lm5, scale=self.face_scale)
                    p = self._infer_gend(aligned)
                    faces.append((bbox, p, True, 'valid'))
                except Exception as e:
                    self.log(f"[GenD] face inference failed: {e}")
                    faces.append((bbox, 0.0, False, 'infer_failed'))
            return faces
        # ---- Haar fallback ----
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ds = min(1.0, 480.0 / W)
        g = cv2.resize(gray, (int(W * ds), int(H * ds))) if ds < 1.0 else gray
        det = self.cascade.detectMultiScale(g, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(40, 40))
        if det is None or len(det) == 0:
            return faces
        boxes = [tuple(float(v) for v in (np.array(f) / ds)) for f in det]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        if max_faces:
            boxes = boxes[:max_faces]
        for (x, y, w, h) in boxes:
            bbox = (x, y, x + w, y + h)
            reason = (assess_face_quality(bbox, None, W, H, self.min_face)
                      if self.quality_filter else 'valid')
            if reason != 'valid':
                faces.append((bbox, 0.0, False, reason))
                continue
            crop = align_face_box(image_bgr, bbox, self.face_scale)
            if crop.size == 0:
                faces.append((bbox, 0.0, False, 'empty_crop'))
                continue
            p = self._infer_gend(crop)
            faces.append((bbox, p, True, 'valid'))
        return faces

    def annotate(self, image_bgr, faces, frame_info=None, show_rejected=True):
        """Annotate boxes (detect_onnx.py annotate + C++ skipped boxes)."""
        vis = image_bgr.copy()
        skip_txt = {'too_small': 'small', 'yaw_too_large': 'side',
                    'pitch_too_large': 'tilt', 'roll_too_large': 'roll'}
        for bbox, p, valid, reason in faces:
            x1, y1, x2, y2 = map(int, bbox[:4])
            if valid:
                color = (0, int(255 * (1 - p)), int(255 * p))
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                text = f"fake: {p:.3f}"
                cv2.putText(vis, text, (x1 + 6, max(20, y1 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, text, (x1 + 6, max(20, y1 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            elif show_rejected:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
                text = skip_txt.get(reason, 'skipped')
                cv2.putText(vis, text, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
        if frame_info:
            cv2.putText(vis, frame_info, (12, vis.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return vis


def get_iframe_positions(video_path, fps, total_frames, gop_seconds=2.0, log=print):
    """I-frame indices (port of C++ IFrameDetector):
    keyframe flags via PyAV -> ffprobe -> GOP estimation fallback."""
    idxs = set()
    try:                                    # 1) PyAV
        import av
        with av.open(video_path) as c:
            s = c.streams.video[0]
            tb = float(s.time_base)
            for pkt in c.demux(s):
                if pkt.pts is None:
                    continue
                if pkt.is_keyframe:
                    idxs.add(int(pkt.pts * tb * fps + 0.5))
        idxs = {i for i in idxs if 0 <= i < total_frames}
    except Exception as e:
        idxs = set()
        log(f"[GenD] PyAV I-frame scan failed: {e}")
    if not idxs:                            # 2) ffprobe
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0',
                   video_path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            for line in r.stdout.splitlines():
                t, key = None, False
                for f in line.split(','):
                    f = f.strip()
                    if 'K' in f:
                        key = True
                    else:
                        try:
                            t = float(f)
                        except ValueError:
                            pass
                if key and t is not None:
                    idxs.add(int(t * fps + 0.5))
            idxs = {i for i in idxs if 0 <= i < total_frames}
        except Exception as e:
            idxs = set()
            log(f"[GenD] ffprobe I-frame scan failed: {e}")
    if not idxs:                            # 3) GOP estimation
        gop = max(1, int(round(gop_seconds * fps)))
        idxs = set(range(0, total_frames, gop))
        log(f"[GenD] I-frame GOP estimation: every {gop} frames (~{gop_seconds:.1f}s)")
    return idxs


# ---- GenD result aggregation (port of C++ FrameResult / VideoResult) ----
def df_state_init(mode):
    return {'mode': mode, 'frame_ids': [], 'frame_avg': [], 'frame_valid': [],
            'frame_rejected': [], 'face_probs': [], 'reject_reasons': {},
            'total_iframes': 0, 'processed_frames': 0,
            'total_valid_faces': 0, 'total_rejected_faces': 0}


def df_state_add(state, frame_id, faces):
    valid_probs = [p for (_, p, ok, _) in faces if ok]
    rejected = [r for (_, _, ok, r) in faces if not ok]
    state['frame_ids'].append(int(frame_id))
    state['frame_valid'].append(len(valid_probs))
    state['frame_rejected'].append(len(rejected))
    state['frame_avg'].append(float(np.mean(valid_probs)) if valid_probs else 0.0)
    state['face_probs'].extend(valid_probs)
    for r in rejected:
        state['reject_reasons'][r] = state['reject_reasons'].get(r, 0) + 1
    state['processed_frames'] += 1
    state['total_valid_faces'] += len(valid_probs)
    state['total_rejected_faces'] += len(rejected)


def _df_snap(state):
    snap = dict(state)
    for k in ('frame_ids', 'frame_avg', 'frame_valid', 'frame_rejected', 'face_probs'):
        snap[k] = list(state[k])
    snap['reject_reasons'] = dict(state['reject_reasons'])
    return snap


def df_state_finalize(state):
    """C++ VideoResult statistics + verdict."""
    fp = state['face_probs']
    pairs = [(a, v) for a, v in zip(state['frame_avg'], state['frame_valid']) if v > 0]
    avgs = [a for a, _ in pairs]
    state['overall_avg_fake_prob'] = float(np.mean(fp)) if fp else 0.0
    state['max_face_prob'] = float(np.max(fp)) if fp else 0.0
    state['min_face_prob'] = float(np.min(fp)) if fp else 0.0
    state['fake_frames'] = sum(1 for a in avgs if a > FAKE_THRESH)
    state['real_frames'] = sum(1 for a in avgs if a <= FAKE_THRESH)
    state['no_face_frames'] = int(sum(1 for v in state['frame_valid'] if v == 0))
    state['max_frame_avg'] = float(np.max(avgs)) if avgs else 0.0
    state['min_frame_avg'] = float(np.min(avgs)) if avgs else 0.0
    avg = state['overall_avg_fake_prob']
    if not fp:
        level, verdict = 'N/A', '[FAKE-N/A] No valid faces for GenD'
    elif avg > FAKE_THRESH:
        level, verdict = 'FAKE', f"[FAKE] Deepfake faces detected (avg {avg:.1%})"
    elif avg > SUSPECT_THRESH:
        level, verdict = 'SUSPECT', f"[SUSPECT] Suspicious faces (avg {avg:.1%})"
    else:
        level, verdict = 'REAL', f"[REAL] Faces look real (avg {avg:.1%})"
    state['level'] = level
    state['verdict'] = verdict
    return state


def combined_verdict(loop_level, df_level, mode='video'):
    """One-line conclusion merging loop result and GenD result."""
    if mode == 'image':
        c = {'FAKE': '#FF5B5B', 'SUSPECT': '#FFB454',
             'REAL': '#3DDC84'}.get(df_level, '#8B9BB4')
        return f"[IMAGE] GenD result: {df_level}", c
    fake = df_level in ('FAKE', 'SUSPECT')
    loop = loop_level in ('LOOP', 'SUSPECT')
    if loop and fake:
        return "[ALERT] DIGITAL HUMAN: motion loop + deepfake face", '#FF5B5B'
    if loop:
        return "[WARN] Loop detected but face looks real (replay of a real person?)", '#FFB454'
    if fake:
        return "[WARN] Deepfake face without obvious motion loop", '#FFB454'
    return "[OK] Natural motion, real face", '#3DDC84'


# ============================== analysis (ported line-by-line from MediaPipe version) ==============================
def _parallel_lag_scores(data, lags_arr, workers):
    def one_chunk(lags):
        return [1.0 / (1.0 + np.mean(np.linalg.norm(data[:-l] - data[l:], axis=1)))
                for l in lags]
    chunks = [c for c in np.array_split(np.asarray(lags_arr, int), workers) if len(c)]
    if not chunks:
        return np.array([]), []
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        parts = list(ex.map(one_chunk, chunks))
    scores = [s for p in parts for s in p]
    return np.array(scores), [int(l) for l in lags_arr]


def analyze_data(data, fps, log=print, max_materials=25):
    """Same structure as LoopAnalyzer.analyze in inference-MultiCore.py (46-dim version).
    data: (N,46) hip-centered feature sequence. returns result dict."""
    n_frames = data.shape[0]
    workers = max(1, min(_NCPU, 8))

    log("Analyzing global periodicity...")
    min_lag = MIN_LAG_FRAMES
    lags_arr = list(range(min_lag, n_frames // 2))
    if len(lags_arr) == 0:
        log(f"[Warn] n_frames={n_frames}: no lag >= {min_lag} frames, "
            f"global periodicity skipped")
        scores = np.array([])
        lags_list = []
        max_sim, best_lag = 0.0, None
    else:
        scores, lags_list = _parallel_lag_scores(data, lags_arr, workers)
        max_sim = float(np.max(scores))
        best_lag = lags_list[int(np.argmax(scores))]   # index -> lag via table (no fixed offset)

    log("Calculating motion entropy...")
    hist, _ = np.histogram(data.flatten(), bins=20)
    prob = hist / max(hist.sum(), 1)
    motion_entropy = _entropy(prob)
    max_entropy = np.log(n_frames * TOTAL_DIM)
    entropy_ratio = motion_entropy / max_entropy if max_entropy > 0 else 0

    log("Detecting multi-peak patterns...")
    if len(scores) == 0:
        peaks = np.array([], int)
    else:
        peaks, properties = find_peaks_compat(scores, height=0.4, distance=15,
                                              prominence=0.08)
    if len(peaks) == 0:
        peak_lags, peak_values = [], []
    else:
        order = np.argsort(scores[peaks])[-10:][::-1]
        peak_lags = [lags_list[peaks[i]] for i in order]
        peak_values = [float(scores[peaks[i]]) for i in order]

    log("Analyzing action clusters...")
    trans_entropy, trans_info = 0.0, {}
    if len(data) < 30:
        n_clusters, cluster_score = 1, 0.0
    else:
        max_k = min(max_materials + 5, len(data) // 10, 30)
        metrics = []
        ks = list(range(2, max_k + 1))

        def one_k(k):
            labels, inertia = kmeans_fit_predict(data, k)
            if len(set(labels.tolist())) < 2:
                return None
            return {'k': k, 'silhouette': silhouette_compat(data, labels),
                    'calinski': calinski_compat(data, labels),
                    'inertia': inertia, 'labels': labels}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for m in ex.map(one_k, ks):
                if m is not None:
                    metrics.append(m)
        if not metrics:
            n_clusters, cluster_score = 1, 0.0
        else:
            best_by_sil = max(metrics, key=lambda x: x['silhouette'])
            n_clusters, cluster_score = best_by_sil['k'], float(best_by_sil['silhouette'])
            cluster_labels = best_by_sil['labels']
            # markov transition
            log("Analyzing transition patterns...")
            if cluster_labels is not None and len(cluster_labels) >= 20:
                uniq = sorted(set(cluster_labels.tolist()))
                sidx = {u: i for i, u in enumerate(uniq)}
                n_states = len(uniq)
                tm = np.zeros((n_states, n_states))
                for a, b in zip(cluster_labels[:-1], cluster_labels[1:]):
                    tm[sidx[a], sidx[b]] += 1
                row_sums = tm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                tp = tm / row_sums
                ents = [_entropy(row[row > 0]) for row in tp if (row > 0).any()]
                trans_entropy = float(np.mean(ents)) if ents else 0.0
                trans_info = {'matrix': tp,
                              'dominant_count': int(np.sum(tp > 0.5)),
                              'avg_entropy': trans_entropy}

    log("Detecting motion transitions...")
    if len(data) < 10:
        jump_indices, n_jumps = np.array([]), 0
    else:
        distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
        jump_threshold = np.mean(distances) + 1.5 * np.std(distances)
        jump_indices = np.where(distances > jump_threshold)[0]
        if len(jump_indices) > 0:
            merged = [int(jump_indices[0])]
            for idx in jump_indices[1:]:
                if idx - merged[-1] > 5:
                    merged.append(int(idx))
            jump_indices = np.array(merged)
        n_jumps = int(len(jump_indices))

    log("Sliding window local periodicity...")

    def one_window(start):
        window = data[start:start + 60]
        n = len(window)
        wl = list(range(5, n // 2))
        sc, _ = _parallel_lag_scores(window, wl, 1)
        if len(sc) == 0:
            return None
        return float(np.max(sc)), start + int(np.argmax(sc)) + 5
    starts = list(range(0, max(len(data) - 60, 0), 30))
    window_scores, window_positions = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one_window, starts):
            if r is not None:
                window_scores.append(r[0])
                window_positions.append(r[1])
    high_periodicity_windows = sum(1 for s in window_scores if s > 0.7)

    multi = {
        'peak_count': len(peak_lags),
        'peak_lags': peak_lags[:5],
        'peak_values': peak_values[:5],
        'cluster_count': n_clusters,
        'cluster_score': cluster_score,
        'jump_count': n_jumps,
        'jump_indices': jump_indices,
        'motion_entropy': motion_entropy,
        'entropy_ratio': float(entropy_ratio),
        'transition_entropy': trans_entropy,
        'transition_info': trans_info,
        'high_periodicity_windows': high_periodicity_windows,
        'window_scores': window_scores,
        'window_positions': window_positions,
    }
    return {'period': best_lag, 'score': max_sim,
            'lags': lags_list, 'scores': scores,
            'multi': multi, 'n_frames': n_frames}


def multi_material_score(multi):
    periodicity_score = multi.get('score', 0) if 'score' in multi else 0
    return None  # placeholder, actual score computed at call site


# ====================== dark tech chart theme (competition UI) ======================
import matplotlib.ticker as mticker

C_BG    = '#0B1220'   # window background
C_PANEL = '#101B2E'   # figure background
C_AX    = '#16223A'   # subplot background
C_GRID  = '#29394F'
C_TXT   = '#E8EEF8'
C_DIM   = '#8B9BB4'
C_CYAN  = '#00D4FF'
C_BLUE  = '#4D8DFF'
C_GREEN = '#3DDC84'
C_AMBER = '#FFB454'
C_RED   = '#FF5B5B'
C_PURP  = '#A78BFA'
C_PINK  = '#FF6EC7'
C_YEL   = '#FFD166'
C_PAL   = [C_CYAN, C_BLUE, C_GREEN, C_AMBER, C_PINK, C_PURP, C_YEL, C_RED]

_THEME_DONE = False


def _apply_chart_theme():
    global _THEME_DONE
    if _THEME_DONE:
        return
    _THEME_DONE = True
    matplotlib.rcParams.update({
        'figure.facecolor': C_PANEL,
        'savefig.facecolor': C_PANEL,
        'savefig.edgecolor': C_PANEL,
        'axes.facecolor': C_AX,
        'axes.edgecolor': C_GRID,
        'axes.labelcolor': C_DIM,
        'xtick.color': C_DIM,
        'ytick.color': C_DIM,
        'text.color': C_TXT,
        'grid.color': C_GRID,
        'grid.alpha': 0.55,
        'font.size': 8.5,
        'legend.facecolor': C_PANEL,
        'legend.edgecolor': C_GRID,
        'legend.framealpha': 0.9,
    })


def _style_ax(ax, title=None, xlabel=None, ylabel=None,
              xint=False, yfmt='%.2f', grid_axis='both', title_color=C_CYAN):
    ax.set_facecolor(C_AX)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_DIM, labelsize=7.5, length=3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6, integer=xint))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    if yfmt:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(yfmt))
    ax.grid(True, axis=grid_axis, color=C_GRID, alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold', color=title_color, pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=C_DIM)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=C_DIM)


def _style_legend(ax, fontsize=6.5, loc='best'):
    leg = ax.legend(fontsize=fontsize, loc=loc)
    if leg is not None:
        for t in leg.get_texts():
            t.set_color(C_TXT)


# ============================== 6-subplot report (2 rows x 3 cols) ==============================
def make_report_figure(fig, result, data, fps):
    _apply_chart_theme()
    fig.clf()
    fig.patch.set_facecolor(C_PANEL)
    multi = result['multi']

    # ---- row 1: feature time series X / Y + right wrist ----
    ax1 = fig.add_subplot(2, 3, 1)
    _style_ax(ax1, 'Feature Time Series (X coordinates)', 'Frame Index', 'Normalized X')
    xs = data[:, 0::2]
    for i in range(xs.shape[1]):
        ax1.plot(xs[:, i], linewidth=0.7, alpha=0.7, color=C_PAL[i % len(C_PAL)])

    ax2 = fig.add_subplot(2, 3, 2)
    _style_ax(ax2, 'Feature Time Series (Y coordinates)', 'Frame Index', 'Normalized Y')
    ys = data[:, 1::2]
    for i in range(ys.shape[1]):
        ax2.plot(ys[:, i], linewidth=0.7, alpha=0.7, color=C_PAL[i % len(C_PAL)])

    ax3 = fig.add_subplot(2, 3, 3)
    _style_ax(ax3, 'Right Wrist Motion', 'Frame Index', 'Normalized Position')
    ax3.plot(data[:, SIGNAL_DIM_X], label='Right Wrist X', linewidth=1.2, color=C_CYAN)
    ax3.plot(data[:, SIGNAL_DIM_Y], label='Right Wrist Y', linewidth=1.2, color=C_AMBER)
    _style_legend(ax3, fontsize=7)

    # ---- row 2: periodicity / analysis results / data quality ----
    ax4 = fig.add_subplot(2, 3, 4)
    _style_ax(ax4, 'Time Lag Periodicity', 'Time Lag (Frames)', 'Similarity Score',
              xint=True, yfmt='%.2f')
    if result['period'] is not None and result['scores'] is not None and len(result['scores']) > 0:
        ax4.plot(result['lags'], result['scores'], linewidth=1.6, color=C_CYAN)
        ax4.axvline(x=result['period'], color=C_RED, linestyle='--', linewidth=1.6,
                    label=f"Primary Period: {result['period']} frames")
        if multi['peak_count'] > 1:
            ax4.axvline(x=multi['peak_lags'][1], color=C_AMBER, linestyle=':',
                        linewidth=1.6, label=f"Secondary: {multi['peak_lags'][1]} frames")
        _style_legend(ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    _style_ax(ax5, 'Analysis Results', 'Score', None, yfmt=None, grid_axis='x')
    periodicity_score = result['score']
    clusters_score = multi['cluster_score']
    transitions_score = 1.0 - min(multi['transition_entropy'] /
                                  np.log(max(multi['cluster_count'], 2)), 1.0)
    jumps_score = 1.0 - min(multi['jump_count'] / 20.0, 1.0)
    entropy_score = 1.0 - multi['entropy_ratio']
    mmat_score = (periodicity_score * 0.4 + clusters_score * 0.25 +
                  transitions_score * 0.15 + entropy_score * 0.1 +
                  jumps_score * 0.1)
    bars = ['Periodicity', 'Clusters', 'Transitions', 'Jumps', 'Entropy', 'Multi-Material']
    vals = [periodicity_score, clusters_score, transitions_score, jumps_score,
            entropy_score, mmat_score]
    colors = [C_CYAN, C_BLUE, C_GREEN, C_AMBER, C_PINK, C_PURP]
    ax5.barh(bars[::-1], vals[::-1], color=colors[::-1], alpha=0.9, height=0.6)
    ax5.set_xlim(0, 1.15)
    for i, val in enumerate(vals[::-1]):
        ax5.text(min(val, 1.0) + 0.025, i, f'{val:.3f}', va='center',
                 fontsize=7, color=C_TXT)

    ax6 = fig.add_subplot(2, 3, 6)
    _style_ax(ax6, 'Data Quality', None, 'Count / %', yfmt='%.0f', grid_axis='y')
    qbars = ['Total\nFrames', 'Valid\nFrames', 'Skipped\nFrames', 'Detection\nRate']
    qvals = [result['total_frames'], result['valid_frames'],
             result['skipped_frames'], result['detection_rate'] * 100]
    qcolors = [C_CYAN, C_GREEN, C_AMBER, C_PURP]
    ax6.bar(range(4), qvals, color=qcolors, alpha=0.9, width=0.6)
    ax6.set_xticks(range(4))
    ax6.set_xticklabels(qbars, fontsize=7)
    top = max(qvals) if max(qvals) > 0 else 1.0
    for i, val in enumerate(qvals):
        ax6.text(i, val + top * 0.03, f'{val:.1f}' if isinstance(val, float) else str(val),
                 ha='center', fontsize=7, color=C_TXT)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.5, wspace=0.35)
    return fig


# ============================== deepfake report (1 row x 4) ==============================
def make_df_figure(fig, state, fps):
    _apply_chart_theme()
    fig.clf()
    fig.patch.set_facecolor(C_PANEL)
    mode = state.get('mode', 'video')
    level = str(state.get('level', 'N/A')).upper()
    lvl_color = (C_RED if 'FAKE' in level else
                 C_AMBER if 'SUSPECT' in level else
                 C_GREEN if 'REAL' in level else C_DIM)

    # 1) fake probability: per I-frame (video) / per face (image)
    ax1 = fig.add_subplot(1, 4, 1)
    if mode == 'image':
        _style_ax(ax1, 'Per-Face Fake Probability', 'Face #', 'P(fake)',
                  xint=True, yfmt='%.1f')
        ax1.set_ylim(0, 1)
        probs = state['face_probs']
        cols = [C_RED if p > FAKE_THRESH else C_CYAN for p in probs]
        ax1.bar(list(range(1, len(probs) + 1)), probs, color=cols, alpha=0.9, width=0.6)
    else:
        _style_ax(ax1, 'Fake Probability per I-Frame', 'Frame Index (I-frames)',
                  'P(fake)', xint=True, yfmt='%.1f')
        ax1.set_ylim(0, 1)
        ax1.plot(state['frame_ids'], state['frame_avg'], 'o-', color=C_CYAN,
                 linewidth=1.3, markersize=3.2)
        if len(state['frame_ids']) > 0:
            ax1.fill_between(state['frame_ids'], state['frame_avg'],
                             alpha=0.12, color=C_CYAN)
        ax1.axhline(FAKE_THRESH, color=C_RED, linestyle='--', linewidth=1.3,
                    label=f'fake {FAKE_THRESH}')
        ax1.axhline(SUSPECT_THRESH, color=C_AMBER, linestyle=':', linewidth=1.3,
                    label=f'suspect {SUSPECT_THRESH}')
        _style_legend(ax1)

    # 2) distribution of face probabilities
    ax2 = fig.add_subplot(1, 4, 2)
    _style_ax(ax2, 'Face Probability Distribution', 'P(fake)', 'Count', yfmt='%.0f')
    probs = state['face_probs']
    if len(probs) > 0:
        ax2.hist(probs, bins=np.linspace(0, 1, 11), color=C_PURP, alpha=0.85,
                 edgecolor=C_PANEL, linewidth=0.8)
        ax2.axvline(np.mean(probs), color=C_RED, linestyle='--', linewidth=1.4,
                    label=f'mean {np.mean(probs):.3f}')
        _style_legend(ax2)
    else:
        ax2.text(0.5, 0.5, 'no valid faces', ha='center', va='center',
                 fontsize=10, color=C_DIM)

    # 3) detection statistics
    ax3 = fig.add_subplot(1, 4, 3)
    _style_ax(ax3, 'Detection Statistics', None, 'Count', yfmt='%.0f', grid_axis='y')
    if mode == 'image':
        labels = ['Faces', 'Valid', 'Rejected']
        vals = [state['total_valid_faces'] + state['total_rejected_faces'],
                state['total_valid_faces'], state['total_rejected_faces']]
    else:
        labels = ['I-Frames', 'Valid\nFaces', 'Rejected\nFaces', 'Fake\nFrames',
                  'Real\nFrames', 'No-Face\nFrames']
        vals = [state['processed_frames'], state['total_valid_faces'],
                state['total_rejected_faces'], state.get('fake_frames', 0),
                state.get('real_frames', 0), state.get('no_face_frames', 0)]
    colors3 = [C_CYAN, C_GREEN, C_AMBER, C_RED, '#2FBF71', C_PURP]
    ax3.bar(labels, vals, color=colors3[:len(labels)], alpha=0.9, width=0.62)
    ax3.tick_params(axis='x', labelsize=7)
    top = max(list(vals) + [1])
    for i, v in enumerate(vals):
        ax3.text(i, v + top * 0.03, str(v), ha='center', fontsize=7, color=C_TXT)

    # 4) conclusion scores
    ax4 = fig.add_subplot(1, 4, 4)
    _style_ax(ax4, f"Conclusion: {state.get('level', 'N/A')}", 'Score', None,
              yfmt=None, grid_axis='x', title_color=lvl_color)
    avg = state.get('overall_avg_fake_prob', 0.0)
    peak = state.get('max_face_prob', 0.0)
    denom = state.get('fake_frames', 0) + state.get('real_frames', 0)
    ratio = (state.get('fake_frames', 0) / denom) if denom else 0.0
    bars = ['Fake Score\n(avg)', 'Peak Risk\n(max)', 'Fake Frame\nRatio', 'Real Score']
    vals4 = [avg, peak, ratio, 1.0 - avg]
    colors4 = [C_RED, C_AMBER, C_PINK, C_GREEN]
    ax4.barh(bars[::-1], vals4[::-1], color=colors4[::-1], alpha=0.9, height=0.58)
    ax4.set_xlim(0, 1.15)
    for i, v in enumerate(vals4[::-1]):
        ax4.text(v + 0.03, i, f'{v:.3f}', va='center', fontsize=7, color=C_TXT)

    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(hspace=0.45, wspace=0.42)
    return fig


# ============================== draw skeleton ==============================
def draw_landmarks(frame, body_norm, body_valid, face_norm, face_valid, lines):
    H, W = frame.shape[:2]
    out = frame
    for a, b in BODY_SKELETON:
        if body_valid[a] and body_valid[b]:
            pa = (int(body_norm[a, 0] * W), int(body_norm[a, 1] * H))
            pb = (int(body_norm[b, 0] * W), int(body_norm[b, 1] * H))
            cv2.line(out, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)
    for i in range(BODY_KPT_NUM):
        if body_valid[i]:
            cv2.circle(out, (int(body_norm[i, 0] * W), int(body_norm[i, 1] * H)),
                       4, (0, 255, 0), -1, cv2.LINE_AA)
    for j in range(FACE_KPT_NUM):
        if face_valid[j]:
            cv2.circle(out, (int(face_norm[j, 0] * W), int(face_norm[j, 1] * H)),
                       3, (255, 0, 0), -1, cv2.LINE_AA)
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        y += 30
    return out


# ============================== pipeline ==============================
def _q_put(q, item, stop_ev):
    while not stop_ev.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


class Pipeline:
    """decode thread -> inference thread (pose NPU || face Haar+NPU || GenD on I-frames)
       -> render/encode thread"""

    def __init__(self, cfg, msg_q, df_det=None, iframe_set=None):
        self.cfg = cfg
        self.msg = msg_q
        self.stop_ev = threading.Event()
        self.q_frames = queue.Queue(maxsize=16)
        self.q_results = queue.Queue(maxsize=16)
        self.q_preview = queue.Queue(maxsize=4)
        self.total_frames = 0
        self.done_frames = 0
        self.sel_frames = 0
        self.start_time = 0.0
        self._analysis_running = False
        self._analysis_lock = threading.Lock()
        self._final_done = threading.Event()
        # ---- GenD deepfake ----
        self.df_det = df_det
        self.iframe_set = set(iframe_set) if iframe_set else set()
        self.df_state = df_state_init('video')
        self.df_state['total_iframes'] = len(self.iframe_set)
        self.df_lock = threading.Lock()
        self._df_count = 0

    # ---------- thread 1: decode ----------
    def _decode_loop(self, cap):
        try:
            idx = 0
            while not self.stop_ev.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if not _q_put(self.q_frames, (idx, frame), self.stop_ev):
                    break
                idx += 1
        finally:
            for _ in range(4):
                self.q_frames.put((None, None))
            self.msg.put(('log', f'[Decode] done: {idx} frames'))

    # ---------- thread 2: inference (pose and face run concurrently) ----------
    def _infer_loop(self, pose_est, face_det, pool):
        try:
            while True:
                idx, frame = self.q_frames.get()
                if idx is None:
                    break
                if self.stop_ev.is_set():
                    continue
                fut_pose = pool.submit(pose_est, frame)
                fut_face = pool.submit(face_det, frame)
                body_norm, body_valid = fut_pose.result()
                pts_px, face_box, n_faces = fut_face.result()
                # business rule: with several people in the frame, analyse only the
                # person with the LARGEST face (body + face must be the same person)
                if n_faces >= 2 and face_box is not None:
                    roi = person_roi_from_face(face_box, frame.shape[1], frame.shape[0])
                    bn, bv = pose_est(frame, roi)
                    if int(bv.sum()) >= MIN_BODY_POINTS:
                        body_norm, body_valid = bn, bv
                        self.sel_frames += 1
                        if self.sel_frames <= 3 or self.sel_frames % 300 == 0:
                            self.msg.put(('log',
                                          f'[Select] frame {idx}: {n_faces} faces -> '
                                          f'target = largest face'))
                # ---- GenD deepfake on I-frames ONLY (CPU) ----
                df_faces = None
                if self.df_det is not None and idx in self.iframe_set:
                    fut_df = pool.submit(self.df_det.detect_frame, frame)
                    df_faces = fut_df.result()
                    with self.df_lock:
                        df_state_add(self.df_state, idx, df_faces)
                        snap = _df_snap(self.df_state)
                    self._df_count += 1
                    if self._df_count <= 3 or self._df_count % 30 == 0:
                        self.msg.put(('log',
                                      f'[GenD] frame {idx} (I-frame): '
                                      f'{snap["frame_valid"][-1]} valid / '
                                      f'{snap["frame_rejected"][-1]} rejected | '
                                      f'avg fake {snap["frame_avg"][-1]:.3f}'))
                    self.msg.put(('dfchart', snap, False))
                face_norm, face_valid = face_to_normalized(pts_px,
                                                           frame.shape[1], frame.shape[0])
                if not _q_put(self.q_results,
                              (idx, frame, body_norm, body_valid,
                               face_norm, face_valid, df_faces),
                              self.stop_ev):
                    break
        finally:
            for _ in range(4):
                self.q_results.put((None,) * 7)
            self.msg.put(('log', '[Inference] done'))

    # ---------- thread 3: render + encode ----------
    def _render_loop(self, writer, analyzer, fps):
        try:
            while True:
                item = self.q_results.get()
                if item[0] is None:
                    break
                idx, frame, body_norm, body_valid, face_norm, face_valid, df_faces = item
                feat = build_frame_features(body_norm, body_valid, face_norm, face_valid)
                analyzer.add_frame_data(feat)

                f5 = int(face_valid.sum())
                lines = [f"Frame: {idx}",
                         f"Body pts: {int(body_valid.sum())}/{BODY_KPT_NUM}",
                         f"Face5 pts: {f5}/5"]
                vis = draw_landmarks(frame, body_norm, body_valid,
                                     face_norm, face_valid, lines)
                if df_faces:
                    vis = self.df_det.annotate(vis, df_faces, f"I-Frame #{idx}")
                writer.write(vis)

                self.done_frames += 1
                # preview (drop frames instead of blocking)
                try:
                    ph, pw = vis.shape[:2]
                    w = PREVIEW_W
                    h = int(w * ph / pw)
                    if h > PREVIEW_MAX_H:      # portrait video: shrink by height
                        h = PREVIEW_MAX_H
                        w = int(h * pw / ph)
                    pv = cv2.resize(vis, (w, h))
                    self.q_preview.put_nowait(pv)
                except queue.Full:
                    pass
                # progress
                if self.done_frames % 10 == 0:
                    elapsed = time.time() - self.start_time
                    speed = self.done_frames / max(elapsed, 1e-6)
                    eta = (self.total_frames - self.done_frames) / max(speed, 1e-6)
                    self.msg.put(('progress', self.done_frames, self.total_frames,
                                  speed, eta))
                # periodic chart refresh
                if self.done_frames % CHART_EVERY == 0:
                    self._maybe_analysis(analyzer, fps, final=False)
        finally:
            writer.release()
            self.msg.put(('log', f'[Done] Output video saved: {self.cfg["output_video"]}'))
            # final analysis
            self._maybe_analysis(analyzer, fps, final=True, block=True)
            if self.df_det is not None:
                with self.df_lock:
                    snap = _df_snap(self.df_state)
                snap = df_state_finalize(snap)
                self.msg.put(('dfchart', snap, True))
            self.msg.put(('finished',))

    # ---------- analysis scheduling ----------
    def _maybe_analysis(self, analyzer, fps, final, block=False):
        with self._analysis_lock:
            if self._analysis_running:
                return
            self._analysis_running = True
        snap = [np.asarray(v, np.float64) for v in analyzer.sequence]
        stats = {'total': analyzer.total_frames, 'valid': analyzer.valid_frames,
                 'skipped': analyzer.skipped_frames}
        t = threading.Thread(target=self._analysis_thread,
                             args=(snap, stats, fps, final), daemon=True)
        t.start()
        if block:
            t.join()

    def _analysis_thread(self, snap, stats, fps, final):
        try:
            if len(snap) < 60:
                self.msg.put(('log', f'[Warn] Not enough valid frames ({len(snap)} < 60), analysis skipped'))
                return
            data = np.array(snap)
            result = analyze_data(data, fps, log=lambda s: self.msg.put(('log', s)))
            rate = stats['valid'] / stats['total'] if stats['total'] else 0
            result.update({'total_frames': stats['total'],
                           'valid_frames': stats['valid'],
                           'skipped_frames': stats['skipped'],
                           'detection_rate': rate})
            self.msg.put(('chart', result, data, final))
        except Exception as e:
            self.msg.put(('error', f'Analysis error: {e}'))
        finally:
            with self._analysis_lock:
                self._analysis_running = False

    # ---------- start ----------
    def start(self, pose_est, face_det, analyzer, cap, writer, fps):
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = time.time()
        pool = ThreadPoolExecutor(max_workers=3)   # pose / face / GenD(I-frame) concurrent
        threads = [
            threading.Thread(target=self._decode_loop, args=(cap,), daemon=True),
            threading.Thread(target=self._infer_loop, args=(pose_est, face_det, pool), daemon=True),
            threading.Thread(target=self._render_loop, args=(writer, analyzer, fps), daemon=True),
        ]
        for t in threads:
            t.start()
        self._threads = threads
        self._pool = pool

    def stop(self):
        self.stop_ev.set()


# ============================== GUI ==============================
class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.msg_q = queue.Queue()
        self.pipeline = None
        self.df_det = None
        self.running = False
        self._photo = None
        self._chart_saved = False
        self._df_chart_saved = False
        self._fps = 30.0
        self._loop_level = 'N/A'
        self._df_level = 'N/A'
        self._df_mode = 'video'
        self._resize_job = None

        root.title("Digital Human Loop Detection - NOE NPU + GenD Deepfake")
        # maximize on 1920x1080 screens (charts auto-fit the right panel)
        try:
            root.state('zoomed')                     # Windows
        except Exception:
            try:
                root.attributes('-zoomed', True)     # Linux/X11
            except Exception:
                root.geometry("1900x1030")

        # ---- top control bar ----
        top = ttk.Frame(root, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text="Video/Image:").grid(row=0, column=0, sticky='w')
        self.var_video = tk.StringVar(value=args.video)
        ttk.Entry(top, textvariable=self.var_video, width=48).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Browse...", command=self.browse).grid(row=0, column=2, padx=4)
        self.btn_start = ttk.Button(top, text="Start", command=self.start)
        self.btn_start.grid(row=0, column=3, padx=4)
        self.btn_stop = ttk.Button(top, text="Stop", command=self.stop, state='disabled')
        self.btn_stop.grid(row=0, column=4, padx=4)

        ttk.Label(top, text="Output video/image:").grid(row=0, column=5, sticky='w', padx=(12, 0))
        self.var_outv = tk.StringVar(value=args.output_video)
        ttk.Entry(top, textvariable=self.var_outv, width=28).grid(row=0, column=6, padx=4)
        ttk.Label(top, text="Output chart:").grid(row=0, column=7, sticky='w', padx=(12, 0))
        self.var_outc = tk.StringVar(value=args.output_chart)
        ttk.Entry(top, textvariable=self.var_outc, width=18).grid(row=0, column=8, padx=4)

        # ---- middle: preview + charts (loop 6 + deepfake 4 merged in one panel) ----
        mid = ttk.Frame(root, padding=8)
        mid.pack(fill='both', expand=True)
        left = ttk.LabelFrame(mid, text="Live Preview", padding=4)
        left.pack(side='left', fill='both', expand=False)
        self.preview_label = ttk.Label(left, text="(select a video/image and click Start)",
                                       anchor='center', background='#222')
        self.preview_label.pack(fill='both', expand=True)

        right = ttk.LabelFrame(mid, text="Analysis (Live)", padding=4)
        right.pack(side='left', fill='both', expand=True, padx=(8, 0))
        # loop report (3 subplot rows) on top, deepfake report (2 rows) below;
        # both figures auto-resize to fill the panel exactly -> no scroll, no blank
        charts = ttk.Frame(right)
        charts.pack(fill='both', expand=True)
        charts.rowconfigure(0, weight=2)
        charts.rowconfigure(1, weight=1)
        charts.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.fig_df = Figure(figsize=(9, 6), dpi=100)
        self.canvas_df = FigureCanvasTkAgg(self.fig_df, master=charts)
        self.canvas_df.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        charts.bind('<Configure>', self._on_charts_resize)
        # [fix] tiny canvas requests -> rows follow weight (2:1 = thirds)
        self.fig.patch.set_facecolor('#0B1220')
        self.fig_df.patch.set_facecolor('#0B1220')
        self.canvas.get_tk_widget().config(width=4, height=4)
        self.canvas_df.get_tk_widget().config(width=4, height=4)

        # ====================== dark tech UI theme (competition) ======================
        C_WIN = '#0B1220'
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('.', background=C_WIN, foreground='#E8EEF8',
                        fieldbackground='#16223A', borderwidth=0)
        style.configure('TFrame', background=C_WIN)
        style.configure('TLabelframe', background=C_WIN, foreground='#00D4FF',
                        bordercolor='#29394F')
        style.configure('TLabelframe.Label', background=C_WIN, foreground='#00D4FF',
                        font=('Helvetica', 10, 'bold'))
        style.configure('TLabel', background=C_WIN, foreground='#E8EEF8')
        style.configure('TEntry', fieldbackground='#16223A', foreground='#E8EEF8',
                        insertcolor='#00D4FF')
        style.configure('TButton', background='#1F2C44', foreground='#E8EEF8',
                        padding=(12, 5), font=('Helvetica', 9, 'bold'))
        style.map('TButton',
                  background=[('active', '#2A3B5C'), ('disabled', '#141C2C')],
                  foreground=[('disabled', '#5A6A82')])
        style.configure('Horizontal.TProgressbar', troughcolor='#0B1220',
                        background='#00D4FF', borderwidth=0)
        self.root.configure(bg=C_WIN)

        # ---- 递归把原生 tk 控件刷成深色（ttk 控件已由 style 覆盖） ----
        def _darken(w):
            cls = w.winfo_class()
            try:
                if cls in ('Frame', 'Toplevel'):
                    w.configure(bg=C_WIN)
                elif cls == 'Label':
                    w.configure(bg=C_WIN, fg='#D5DEEB')
                elif cls == 'Button':
                    w.configure(bg='#1F2C44', fg='#E8EEF8',
                                activebackground='#2A3B5C', activeforeground='#E8EEF8')
                elif cls == 'Canvas':
                    w.configure(bg=C_WIN, highlightthickness=0)
            except Exception:
                pass
            for c in w.winfo_children():
                _darken(c)

        _darken(self.root)
        try:
            self.preview_label.configure(bg=C_WIN, fg='#5C6B84')
        except Exception:
            pass
        try:
            self.log_text.configure(bg=C_WIN, fg='#9EC3E8', insertbackground='#00D4FF',
                                    selectbackground='#2A5A8C', selectforeground='#E8EEF8',
                                    highlightthickness=0, relief='flat')
        except Exception:
            pass
        for cvs in (self.canvas, self.canvas_df):
            try:
                cvs.get_tk_widget().configure(bg='#101B2E', highlightthickness=0)
            except Exception:
                pass

        # ---- bottom: progress + verdicts + log ----
        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill='x')
        self.progress = ttk.Progressbar(bottom, maximum=1000)
        self.progress.pack(fill='x', pady=2)
        self.var_status = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.var_status).pack(anchor='w')
        self.var_verdict = tk.StringVar(value="")
        self.lbl_verdict = ttk.Label(bottom, textvariable=self.var_verdict,
                                     font=('Helvetica', 14, 'bold'))
        self.lbl_verdict.pack(anchor='w', pady=2)
        self.var_df_verdict = tk.StringVar(value="")
        self.lbl_df_verdict = ttk.Label(bottom, textvariable=self.var_df_verdict,
                                        font=('Helvetica', 11, 'bold'))
        self.lbl_df_verdict.pack(anchor='w', pady=2)
        self.var_combined = tk.StringVar(value="")
        self.lbl_combined = ttk.Label(bottom, textvariable=self.var_combined,
                                      font=('Helvetica', 13, 'bold'))
        self.lbl_combined.pack(anchor='w', pady=2)
        self.log_text = tk.Text(bottom, height=8, font=('Consolas', 9))
        self.log_text.pack(fill='x', pady=2)
        # [fix] log_text dark (theme pass ran before it was created)
        try:
            self.log_text.configure(bg=C_WIN, fg='#9EC3E8',
                                    insertbackground='#00D4FF',
                                    selectbackground='#2A5A8C',
                                    selectforeground='#E8EEF8',
                                    highlightthickness=0, relief='flat')
        except Exception:
            pass

        self.root.after(120, self.poll)

    # ---------- chart auto-fit (fill the right panel exactly) ----------
    def _on_charts_resize(self, event=None):
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(150, self._do_resize_charts)

    def _do_resize_charts(self):
        self._resize_job = None
        try:
            w1 = self.canvas.get_tk_widget().winfo_width()
            h1 = self.canvas.get_tk_widget().winfo_height()
            w2 = self.canvas_df.get_tk_widget().winfo_width()
            h2 = self.canvas_df.get_tk_widget().winfo_height()
            if min(w1, h1, w2, h2) < 50:            # layout not ready yet
                self.root.after(200, self._do_resize_charts)
                return
            dpi = self.fig.dpi
            self.fig.set_size_inches(w1 / dpi, h1 / dpi, forward=False)
            self.canvas.draw_idle()
            self.fig_df.set_size_inches(w2 / dpi, h2 / dpi, forward=False)
            self.canvas_df.draw_idle()
        except Exception:
            pass

    # ---------- file selection ----------
    def browse(self):
        p = filedialog.askopenfilename(
            title="Select video or image",
            filetypes=[("Video/Image", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All", "*.*")])
        if p:
            self.var_video.set(p)

    # ---------- start ----------
    def start(self):
        if self.running:
            return
        path = self.var_video.get().strip()
        try:
            require_file(path)
        except Exception as e:
            self.log(f"[ERROR] {e}")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTS:
            self.start_image_mode(path)
            return

        try:
            require_file(self.args.pose_model)
            require_file(self.args.face_model)
        except Exception as e:
            self.log(f"[ERROR] {e}")
            return

        self.log("=" * 50)
        self.log(f"[Model] loading: {self.args.pose_model} / {self.args.face_model}")
        try:
            pose_model = NoeModel(self.args.pose_model)
            face_model = NoeModel(self.args.face_model)
        except Exception as e:
            self.log(f"[ERROR] Model load failed: {e}")
            return
        pose_est = PoseEstimator(pose_model)
        face_det = FaceLandmarkDetector(face_model)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.log(f"[ERROR] Cannot open video: {path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.log(f"   Resolution {W}x{H} | FPS {fps:.2f} | Total frames {total}")

        writer = cv2.VideoWriter(self.var_outv.get(),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        if not writer.isOpened():
            self.log("[ERROR] Cannot create output video")
            return

        # ---- GenD deepfake (CPU) on I-frames ----
        self.df_det = None
        iframe_set = None
        if not self.args.no_gend:
            try:
                self.df_det = GendDeepfakeDetector(
                    self.args.gend_model, self.args.det_model,
                    gend_root=self.args.gend_root,
                    det_thresh=self.args.det_thresh,
                    face_scale=self.args.face_scale,
                    min_face=self.args.min_face,
                    quality_filter=not self.args.no_quality_filter,
                    log=self.log)
                iframe_set = get_iframe_positions(path, fps, total,
                                                  gop_seconds=self.args.gend_gop,
                                                  log=self.log)
                self.log(f"[GenD] I-frame detection enabled: "
                         f"{len(iframe_set)} keyframes to process")
            except Exception as e:
                self.log(f"[WARN] GenD disabled: {e}")
                self.df_det = None
                iframe_set = None

        analyzer = LoopAnalyzer()
        self.pipeline = Pipeline({'output_video': self.var_outv.get(),
                                  'output_chart': self.var_outc.get()},
                                 self.msg_q, self.df_det, iframe_set)
        self._fps = fps
        self._chart_saved = False
        self._df_chart_saved = False
        self._loop_level = 'N/A'
        self._df_level = 'N/A'
        self._df_mode = 'video'
        self.running = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.var_verdict.set("")
        self.var_df_verdict.set("")
        self.var_combined.set("")
        self.pipeline.start(pose_est, face_det, analyzer, cap, writer, fps)
        self.log("[Start] processing (3-stage pipeline: decode | infer | render+encode)")

    # ---------- image mode: GenD only ----------
    def start_image_mode(self, path):
        self.log("=" * 50)
        self.log(f"[Mode] Image input -> GenD deepfake only: {path}")
        try:
            self.df_det = GendDeepfakeDetector(
                self.args.gend_model, self.args.det_model,
                gend_root=self.args.gend_root,
                det_thresh=self.args.det_thresh,
                face_scale=self.args.face_scale,
                min_face=self.args.min_face,
                quality_filter=False,
                log=self.log)
        except Exception as e:
            self.log(f"[ERROR] GenD unavailable: {e}")
            return
        img = cv2.imread(path)
        if img is None:
            self.log(f"[ERROR] Cannot read image: {path}")
            return
        self._fps = 30.0
        self._chart_saved = False
        self._df_chart_saved = False
        self._loop_level = 'N/A'
        self._df_level = 'N/A'
        self._df_mode = 'image'
        self.running = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.var_verdict.set("")
        self.var_df_verdict.set("")
        self.var_combined.set("")
        # loop figure placeholder (image mode has no loop analysis)
        _apply_chart_theme()
        self.fig.clf()
        self.fig.patch.set_facecolor(C_PANEL)
        ax = self.fig.add_subplot(111)
        ax.axis('off')
        ax.set_title('Image mode: loop analysis not applicable', fontsize=11,
                     color=C_DIM)
        self.canvas.draw_idle()
        threading.Thread(target=self._image_worker, args=(path, img),
                         daemon=True).start()

    def _image_worker(self, path, img):
        try:
            faces = self.df_det.detect_frame(img)
            state = df_state_init('image')
            df_state_add(state, 0, faces)
            state['total_iframes'] = 1
            state = df_state_finalize(state)
            vis = self.df_det.annotate(img, faces, os.path.basename(path))
            out_img = self.var_outv.get()
            if os.path.splitext(out_img)[1].lower() not in IMAGE_EXTS:
                out_img = os.path.splitext(out_img)[0] + '_gend.jpg'
            cv2.imwrite(out_img, vis)
            self.msg_q.put(('log', f"[GenD] faces: {state['total_valid_faces']} valid / "
                                   f"{state['total_rejected_faces']} rejected"))
            self.msg_q.put(('log', f"[GenD] Annotated image saved: {out_img}"))
            self.msg_q.put(('dfimage', vis))
            self.msg_q.put(('dfchart', state, True))
        except Exception as e:
            self.msg_q.put(('error', f'GenD image error: {e}'))
        finally:
            self.msg_q.put(('finished',))


    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
            self.log("[Stop] requested, draining...")

    # ---------- message polling ----------
    def poll(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]
                if kind == 'log':
                    self.log(msg[1])
                elif kind == 'progress':
                    _, done, total, speed, eta = msg
                    self.progress['value'] = 1000 * done / max(total, 1)
                    self.var_status.set(
                        f"Progress {done}/{total} ({100 * done / max(total, 1):.1f}%) | "
                        f"{speed:.2f} FPS | ETA {eta:.0f}s")
                elif kind == 'chart':
                    _, result, data, final = msg
                    self.draw_chart(result, data, final)
                elif kind == 'dfchart':
                    self.draw_df_chart(msg[1], msg[2])
                elif kind == 'dfimage':
                    self.show_preview(msg[1])
                elif kind == 'error':
                    self.log(f"[ERROR] {msg[1]}")
                elif kind == 'finished':
                    self.running = False
                    self.btn_start.config(state='normal')
                    self.btn_stop.config(state='disabled')
                    self.var_status.set("Done")
        except queue.Empty:
            pass
        # preview frame
        try:
            frame = self.q_preview_latest()
            if frame is not None and HAS_PIL:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._photo = ImageTk.PhotoImage(img)
                self.preview_label.config(image=self._photo, text="")
        except Exception:
            pass
        self.root.after(120, self.poll)

    def q_preview_latest(self):
        if not self.pipeline:
            return None
        frame = None
        try:
            while True:
                frame = self.pipeline.q_preview.get_nowait()
        except queue.Empty:
            return frame

    def show_preview(self, frame_bgr):
        if not HAS_PIL:
            return
        ph, pw = frame_bgr.shape[:2]
        w = PREVIEW_W
        h = int(w * ph / pw)
        if h > PREVIEW_MAX_H:
            h = PREVIEW_MAX_H
            w = int(h * pw / ph)
        img = Image.fromarray(cv2.cvtColor(cv2.resize(frame_bgr, (w, h)),
                                           cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._photo, text="")

    # ---------- verdicts ----------
    def update_combined(self):
        text, color = combined_verdict(self._loop_level, self._df_level, self._df_mode)
        self.var_combined.set(text)
        self.lbl_combined.config(foreground=color)

    def draw_chart(self, result, data, final):
        make_report_figure(self.fig, result, data, self._fps)
        self.canvas.draw_idle()
        score = result['score']
        period = result['period']
        rate = result['detection_rate']
        if score > 0.85 and rate > 0.7:
            self._loop_level = 'LOOP'
            verdict = f"[LOOP] High-repetition loop | period {period} frames | similarity {score:.4f}"
            color = '#FF5B5B'
        elif score > 0.70 and rate > 0.5:
            self._loop_level = 'SUSPECT'
            verdict = f"[SUSPECT] Suspicious periodicity | period {period} frames | similarity {score:.4f}"
            color = '#FFB454'
        else:
            self._loop_level = 'OK'
            verdict = f"[OK] Natural motion, no clear loop | similarity {score:.4f}"
            color = '#3DDC84'
        self.var_verdict.set(verdict)
        self.lbl_verdict.config(foreground=color)
        self.update_combined()
        if final and not self._chart_saved:
            out = self.var_outc.get()
            self.fig.savefig(out, dpi=150, bbox_inches='tight')
            self._chart_saved = True
            self.log(f"[Saved] Analysis chart: {out}")
            m = result['multi']
            psec = (period / self._fps) if period is not None else 0.0
            self.log(f"[Result] Period: {period} frames ({psec:.2f}s) | "
                     f"Similarity: {score:.4f} | Peaks: {m['peak_count']} | "
                     f"Clusters: {m['cluster_count']} | Jumps: {m['jump_count']} | "
                     f"Entropy: {m['motion_entropy']:.3f} | "
                     f"Trans: {m['transition_entropy']:.3f} | "
                     f"HighWin: {m['high_periodicity_windows']}")

    def draw_df_chart(self, state, final):
        make_df_figure(self.fig_df, state, self._fps)
        self.canvas_df.draw_idle()
        self._df_mode = state.get('mode', 'video')
        self._df_level = state.get('level', 'N/A')
        self.var_df_verdict.set(state.get('verdict', ''))
        self.lbl_df_verdict.config(
            foreground={'FAKE': '#FF5B5B', 'SUSPECT': '#FFB454',
                        'REAL': '#3DDC84'}.get(self._df_level, '#8B9BB4'))
        self.update_combined()
        if final and not self._df_chart_saved:
            out = self.var_outc.get()
            base, ext = os.path.splitext(out)
            out_df = base + '_df' + (ext or '.png')
            self.fig_df.savefig(out_df, dpi=150, bbox_inches='tight')
            self._df_chart_saved = True
            self.log(f"[Saved] Deepfake chart: {out_df}")
            self.log(f"[GenD] I-frames: {state['processed_frames']}"
                     f"/{state.get('total_iframes', 0)} | "
                     f"Valid faces: {state['total_valid_faces']} | "
                     f"Rejected: {state['total_rejected_faces']} | "
                     f"Avg fake: {state.get('overall_avg_fake_prob', 0.0):.3f} | "
                     f"Max: {state.get('max_face_prob', 0.0):.3f} | "
                     f"Fake frames: {state.get('fake_frames', 0)} | "
                     f"Real frames: {state.get('real_frames', 0)}")
            if state.get('reject_reasons'):
                rs = ', '.join(f'{k}:{v}' for k, v in state['reject_reasons'].items())
                self.log(f"[GenD] Rejected reasons: {rs}")

    def log(self, s):
        self.log_text.insert('end', s + '\n')
        self.log_text.see('end')
        print(s)


# ============================== LoopAnalyzer (collector) ==============================
class LoopAnalyzer:
    def __init__(self):
        self.sequence = []
        self.total_frames = 0
        self.valid_frames = 0
        self.skipped_frames = 0

    def add_frame_data(self, feat):
        self.total_frames += 1
        if feat is None:
            self.skipped_frames += 1
            return
        self.valid_frames += 1
        self.sequence.append(feat)


# ============================== main ==============================
def parse_args():
    p = argparse.ArgumentParser(description="Digital Human Loop Detection (NOE NPU GUI + GenD)")
    p.add_argument("--video", default="test.mp4",
                   help="video (loop + I-frame GenD) or image (GenD only)")
    p.add_argument("--pose_model", default="model/human-pose-estimation.cix")
    p.add_argument("--face_model", default="model/facenet.cix")
    p.add_argument("--output_video", default="output_visualized.mp4",
                   help="output video / annotated image")
    p.add_argument("--output_chart", default="loop_analysis.png")
    # ---- GenD deepfake (CPU) ----
    p.add_argument("--gend_model", default="model/gend.onnx")
    p.add_argument("--det_model", default="model/det_10g.onnx")
    p.add_argument("--gend_root", default=".",
                   help="(unused, kept for backward compat; face backend is bundled)")
    p.add_argument("--det_thresh", type=float, default=0.5)
    p.add_argument("--face_scale", type=float, default=1.3)
    p.add_argument("--min_face", type=int, default=80)
    p.add_argument("--max_faces", type=int, default=0, help="0 = unlimited")
    p.add_argument("--gend_gop", type=float, default=2.0,
                   help="fallback I-frame interval in seconds")
    p.add_argument("--no_gend", action='store_true', help="disable GenD deepfake")
    p.add_argument("--no_quality_filter", action='store_true',
                   help="disable face quality filter (size/yaw/pitch/roll)")
    return p.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
