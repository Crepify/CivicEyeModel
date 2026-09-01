# CivicEye Model

ONNX models for **CivicEye** — a YOLO-based civic-issue detector (potholes, garbage, manholes, etc.).

Model graph: `input: images [1, 3, 640, 640] (float32)` → `output [1, 15, 8400]`
(8400 anchors, each = `cx, cy, w, h` + **11 class scores**). Exported from PyTorch (fp16), opset 17.

> Note: the fp16 export names its output `output0`; the quantized int8 names it `graph_output_cast_0`.
> It doesn't matter — `onDeviceYolo.ts` reads `Object.values(results)[0]`, i.e. the first output by value.

---

## 📦 Files

| File | Size | Format | Where to use it |
|---|---|---|---|
| `models/civiceye.onnx` | 25.1 MB | FP16 (original, exact export) | Full precision. Browser-compatible (works in onnxruntime-web). **Too big for jsDelivr** — host on Hugging Face / GitHub release if you need it served. |
| `models/civiceye-int8.onnx` | 12.5 MB | INT8 static quantization (QDQ) | **jsDelivr-ready** (under the 20 MB limit) and validated end-to-end with `onnxruntime-web` WASM (session + inference both pass). ~0.8% output drift vs FP16 — visually identical detections. |

Both models are interchangeable: same input name `images`, same output layout `[1, 15, 8400]`.

---

## 🌐 jsDelivr URLs (live — verified)

```
https://cdn.jsdelivr.net/gh/Crepify/CivicEyeModel@v1.1.0/models/civiceye-int8.onnx
```

- ✅ Serving live with CORS enabled (`access-control-allow-origin: *`), byte-identical to the pushed file.
- **Use the `@v1.1.0` tagged URL** (a fresh tag = fresh content on jsDelivr, no cache staleness). Update the tag (`v1.1.0`, etc.) whenever you push a new model.
- jsDelivr only serves files **≤ 20 MB** and does **not** support Git LFS — that's why the quantized file is the one served (`civiceye.onnx` at 25 MB returns 403 from jsDelivr, as expected; host it on Hugging Face if you need the full-precision version served).

---

## 🚀 Wiring into the CivicEye app

Your `src/services/onDeviceYolo.ts` already supports a custom ONNX YOLO via env vars:

```bash
VITE_ONDEVICE_YOLO_URL=https://cdn.jsdelivr.net/gh/Crepify/CivicEyeModel@v1.1.0/models/civiceye-int8.onnx
VITE_ONDEVICE_YOLO_LABELS=pothole,garbage,manhole,street-light,fallen-tree,sewage,water-leakage,broken-road,sidewalk,illegal-dumping,other
VITE_ONDEVICE_YOLO_SIZE=640
VITE_ONDEVICE_YOLO_CONF=0.35
```

> ⚠️ **Set `VITE_ONDEVICE_YOLO_LABELS` to your Roboflow training classes in their exact export order**
> (the `classes.txt` / `data.yaml` from your Roboflow export zip). The model has **11 classes**; wrong order = wrong labels on boxes.

**Coordinate-space caveat:** this model outputs box coordinates in **0–640 pixel space**
(verified empirically: box values range up to ~636). `decodeYoloOutput()` in `onDeviceYolo.ts`
currently multiplies boxes by `imgW/imgH`, which assumes **normalized (0–1)** coords.
If boxes render oversized, map pixel→original image with `scale`/`pad` instead, e.g.:

```ts
const x1 = (cx - w / 2 - padX) / scale;
const y1 = (cy - h / 2 - padY) / scale;
const x2 = (cx + w / 2 - padX) / scale;
const y2 = (cy + h / 2 - padY) / scale;
```

(`scale`, `padX`, `padY` are already returned by `preprocess()`.)

---

## 🧪 Test it

Open `demo/index.html` in any browser (double-click it, or serve the folder):

1. Pick/upload a photo
2. Adjust labels if needed
3. Click **Detect** → boxes are drawn with class + confidence

The demo loads the model straight from the jsDelivr URL and uses the same WASM runtime
(`onnxruntime-web`) as the app — it's also a quick way to check your label order.

---

## 🔁 Re-quantizing (if you retrain)

```bash
pip install onnxruntime pillow
python - <<'EOF'
import glob
from PIL import Image
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

class Reader(CalibrationDataReader):
    def __init__(self, files):
        self.data = []
        for f in files:
            im = Image.open(f).convert('RGB')
            im = im.resize((640, 640))          # or letterbox
            a = np.asarray(im).astype(np.float32) / 255.0
            self.data.append({'images': a.transpose(2, 0, 1)[None]})
        self.i = 0
    def get_next(self):
        if self.i < len(self.data):
            d = self.data[self.i]; self.i += 1
            return d

quantize_static('civiceye.onnx', 'civiceye-int8.onnx', Reader(glob.glob('calib/*.jpg')),
                quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
EOF
```

Use **QDQ (QuantizeLinear/DequantizeLinear) static** quantization only — dynamic/`ConvInteger`
quantization is NOT supported by `onnxruntime-web`, and neither is Git LFS on jsDelivr.

---

## 📏 Size limits cheat-sheet

| | GitHub | jsDelivr |
|---|---|---|
| Per-file limit | 100 MB | 20 MB |
| Git LFS | ✓ | ✗ (serves the pointer file only) |
