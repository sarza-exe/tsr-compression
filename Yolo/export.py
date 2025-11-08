import torch
from ultralytics import YOLO
from torch.utils.mobile_optimizer import optimize_for_mobile
import os

MODEL_PT_PATH = 'runs/detect/yolov8n_gtsdb_416/weights/best.pt'
TEMP_TS_PATH = 'runs/detect/yolov8n_gtsdb_416/weights/best.torchscript'
FINAL_PTL_PATH = 'gtsdb_yolo_416.ptl'
INPUT_SIZE = 416

def export_without_nms():
    if not os.path.exists(MODEL_PT_PATH):
        raise FileNotFoundError(MODEL_PT_PATH)

    model_ul = YOLO(MODEL_PT_PATH)

    # Export WITHOUT NMS: model zwróci surowe predykcje
    model_ul.export(format='torchscript', imgsz=INPUT_SIZE, nms=False, optimize=False)

def make_ptl_from_ts(ts_path, out_path):
    model_ts = torch.jit.load(ts_path).eval()
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = model_ts(example)
    print("DEBUG: example output type:", type(out))
    try:
        print("DEBUG: output.shape:", out.shape)
    except Exception:
        print("DEBUG: output (non-tensor) ->", out)

    # Optymalizuj i zapisz .ptl
    optimized = optimize_for_mobile(model_ts)
    optimized._save_for_lite_interpreter(out_path)
    print("Saved ptl to", out_path)

if __name__ == "__main__":
    export_without_nms()
    ts_path = TEMP_TS_PATH
    if not os.path.exists(ts_path):
        # print files in dir to help locate
        print("TorchScript file not found at", ts_path)
        print("Check runs/detect/your_run/weights/")
    else:
        make_ptl_from_ts(ts_path, FINAL_PTL_PATH)
