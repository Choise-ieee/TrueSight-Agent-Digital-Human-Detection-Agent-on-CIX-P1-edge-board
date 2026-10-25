# TrueSight-Agent-Digital-Human-Detection-Agent-on-CIX-P1-edge-board
Real-time digital human content audit on the CIX (此芯科技) P1 edge board: detects loop-replay fraud (recorded footage posing as live) and face-swap (deepfake) content, in a single-file Tkinter GUI with a live single-screen dashboard.


**One Python file · offline · ≈ 9 FPS (≈ 111 ms/frame) · video never leaves the device**

<img width="3088" height="1708" alt="image" src="https://github.com/user-attachments/assets/4635cc99-0761-4c49-b41d-b6170d1abf41" />



> 2026 CIX Agentic AI Developer Contest「芯智涌现」· Edge On-Device Inference Agent  
> Team **幻觉和他的朋友们** — Zhang Lin · Xu Huaijian · Lu Linfeng

## Highlights

| | |
|---|---|
| **Dual engines** | motion-loop (NPU pose → lag periodicity + 6-dim score) + face-swap (GenD / CLIP ViT-L/14) |
| **I-frame sparse** | classify keyframes only (GOP ≈ 2 s) — ~10× cheaper, same accuracy |
| **Heterogeneous** | CNN hot path on **NPU** (pose/facenet/det_10g, 5.3 ms) · ViT judgment on **CPU** · GPU hard-decode on roadmap |
| **Verdict** | three lines: loop / fake / merged, e.g. `[LOOP+FAKE] Replayed video with swapped faces` |

## Results (on board)

| Metric | Value |
|---|---|
| End-to-end | **≈ 111 ms/frame (9 FPS)** · target was &lt; 200 ms |
| Face-swap A/B | **0.391 / 0.985** (= PC 0.392 / 0.985, zero drift) |
| Loop baseline | 965 frames · peak 0.9795 · 98% / 100% |
| Pose NPU forward | **5.3 ms** |

## Quick start

```bash
pip install -r requirements.txt
sudo apt install python3-tk
python3 pose_loop_npu_gui.py --video your.mp4
```

Python 3.11 · Linux · CIX P1 / NOE SDK (`libnoe`). Image input runs GenD only.

## Models

Weights are **not** in the repo — see “https://drive.google.com/file/d/1_pUA6jvAeOf6qu7DfQ-X-dQFu-2MJhiQ/view?usp=drive_link”.

| File | Backend | Task |
|---|---|---|
| `human-pose-estimation.cix` / `facenet.cix` / `det_10g.cix` | NPU | pose / embed / boxes |
| `det_10g.onnx` · `gend.onnx` | CPU (onnxruntime) | face crop · GenD |


## Heterogeneous scheduling

CNN goes to NPU; GenD stays on CPU **by design** — face-swap needs fine logit gaps that PTQ cannot hold (7 conversion rounds verified; detection heads are fine on the same chain). I-frame rate ≈ 0.5/s, so CPU cost is negligible. Roadmap: **CPU + NPU + GPU**.


