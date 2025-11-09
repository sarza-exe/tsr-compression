"""
 - generuje plik .ptl dla PyTorch Mobile Lite, bez zależności torchvision::nms,
 - NMS napisany w czystym PyTorch,
 - zwraca tensor (N,6) z [x1,y1,x2,y2,score,class] gdzie x/y są znormalizowane do [0..1] względem INPUT_SIZE.

Użycie:
 python exportNMS.py (wtedy szuka .torchscript domyślnie w runs/detect/yolov8n_gtsdb_416/weights)
 python exportNMS.py --model_pt runs/detect/.../weights/best.pt --out_ptl gtsdb_yolo_416.ptl --input_size 416

Jeśli masz TorchScript wyeksportowany przez Ultralytics (bez nms), możesz podać --ts_path .
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import List

# ---------------------------
# Scriptable pure-torch NMS
# ---------------------------
@torch.jit.script
def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_th: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep: List[int] = torch.jit.annotate(List[int], [])
    while order.numel() > 0:
        first_idx_tensor = order[0:1]
        keep.append(int(first_idx_tensor[0].item()))
        if order.numel() == 1:
            break
        cur_box = boxes.index_select(0, first_idx_tensor)   # (1,4)
        rest_idx = order[1:]
        rest = boxes.index_select(0, rest_idx)              # (M,4)
        xx1 = torch.max(cur_box[:, 0], rest[:, 0])
        yy1 = torch.max(cur_box[:, 1], rest[:, 1])
        xx2 = torch.min(cur_box[:, 2], rest[:, 2])
        yy2 = torch.min(cur_box[:, 3], rest[:, 3])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        area_cur = (cur_box[:, 2] - cur_box[:, 0]).clamp(min=0) * (cur_box[:, 3] - cur_box[:, 1]).clamp(min=0)
        area_rest = (rest[:, 2] - rest[:, 0]).clamp(min=0) * (rest[:, 3] - rest[:, 1]).clamp(min=0)
        union = area_cur + area_rest - inter
        ious = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        mask = ious <= iou_th
        order = order[1:][mask]
    if len(keep) == 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(keep, dtype=torch.long)

class WrapperModule(nn.Module):
    def __init__(self, model, input_size:int = 416, conf_thresh:float = 0.25, iou_thresh:float = 0.45):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def forward(self, x):
        # x: (1,3,H,W)
        preds = self.model(x)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # preds expected: tensor (1,5,3549) as in your dump
        if preds.dim() == 3 and preds.size(1) == 5:
            # squeeze batch -> (5, 3549) then transpose -> (3549, 5)
            p = preds.squeeze(0).permute(1, 0)  # (num_preds, 5)
            # split channels
            cx = p[:, 0]
            cy = p[:, 1]
            pw = p[:, 2]
            ph = p[:, 3]
            obj = p[:, 4]

            # convert center->xyxy (absolute pixels)
            x1 = cx - pw / 2.0
            y1 = cy - ph / 2.0
            x2 = cx + pw / 2.0
            y2 = cy + ph / 2.0

            boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N,4)
            scores = obj  # objectness as score
            classes = torch.zeros_like(scores)  # single-class detector -> class 0

        else:
            # fallback: try to reuse the previous generic decoder (handles 1,N,6 or N,C)
            # This keeps compatibility with other exports: (reuse logic from earlier wrapper)
            p = preds
            if p.dim() == 3 and p.size(0) == 1:
                p = p.squeeze(0)
            # If p has last dim 6 treat as [x1,y1,x2,y2,score,class]
            if p.dim() == 2 and p.size(1) == 6:
                boxes = p[:, :4]
                scores = p[:, 4]
                classes = p[:, 5]
            else:
                # unknown: return empty
                return p.new_zeros((0,6))

        # filter by conf thresh
        keep_mask = scores > self.conf_thresh
        if keep_mask.sum() == 0:
            return boxes.new_zeros((0,6))

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        classes = classes[keep_mask]

        # Ensure boxes are absolute pixels for NMS.
        # Heuristic: if max(boxes) <= 1 -> assume normalized, multiply by input_size
        if boxes.numel() > 0 and boxes.max() <= 1.0:
            boxes_abs = boxes * float(self.input_size)
        else:
            boxes_abs = boxes

        keep_idx = nms_torch(boxes_abs, scores, float(self.iou_thresh))
        if keep_idx.numel() == 0:
            return boxes.new_zeros((0,6))

        kept_boxes = boxes_abs[keep_idx]
        kept_scores = scores[keep_idx]
        kept_classes = classes[keep_idx]

        # Normalize boxes to [0..1] relative to input_size for mobile output
        out = torch.stack([
            kept_boxes[:,0] / float(self.input_size),
            kept_boxes[:,1] / float(self.input_size),
            kept_boxes[:,2] / float(self.input_size),
            kept_boxes[:,3] / float(self.input_size),
            kept_scores,
            kept_classes
        ], dim=1)

        return out  # (M,6)


def find_torchscript_in_runs(runs_root="runs/detect/yolov8n_gtsdb_416/weights"):
    # Try to find a .torchscript file in runs/detect subfolders
    for root, dirs, files in os.walk(runs_root):
        for f in files:
            if f.endswith(".torchscript") or f.endswith(".torchscript.pt") or f.endswith(".ts"):
                return os.path.join(root, f)
    return None


def main(args):
    ts_path = args.ts_path
    if ts_path is None or not os.path.exists(ts_path):
        print("[info] Provided TorchScript not found.")
        if args.model_pt and os.path.exists(args.model_pt):
            # use ultralytics to export (if available)
            try:
                from ultralytics import YOLO
            except Exception as e:
                print("[error] ultralytics not available; please provide '--ts_path' or install ultralytics.")
                raise

            print("[info] Exporting Ultralytics model to TorchScript (nms=False)... this may take a moment.")
            model_ul = YOLO(args.model_pt)
            # export to torchscript without torchvision NMS
            model_ul.export(format='torchscript', imgsz=args.input_size, nms=False, optimize=False)
            # try to find generated .torchscript
            ts_path_guess = find_torchscript_in_runs()
            if ts_path_guess:
                print("[info] Found torchscript at:", ts_path_guess)
                ts_path = ts_path_guess
            else:
                raise FileNotFoundError("TorchScript export not found. Provide --ts_path or inspect runs/detect/.../weights/")
        else:
            # try to find in runs/detect
            ts_guess = find_torchscript_in_runs()
            if ts_guess:
                print("[info] Found torchscript at:", ts_guess)
                ts_path = ts_guess
            else:
                raise FileNotFoundError("No TorchScript provided and cannot export. Provide --ts_path or --model_pt.")

    print("[info] Loading TorchScript:", ts_path)
    ts_model = torch.jit.load(ts_path)
    ts_model.eval()

    # quick inspect: print shape on example
    example = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        try:
            out = ts_model(example)
            print("[debug] raw model output type:", type(out))
            try:
                if torch.is_tensor(out):
                    print("[debug] raw output shape:", tuple(out.shape))
                else:
                    # list/tuple
                    print("[debug] raw output len/list:", len(out))
            except Exception:
                pass
        except Exception as e:
            print("[warning] Running example through raw ts_model failed:", e)
            # continue; may still be scriptable wrapper

    # Wrap with our WrapperModule and script it
    print("[info] Wrapping model with scriptable NMS wrapper...")
    wrapper = WrapperModule(ts_model, input_size=args.input_size, conf_thresh=args.conf, iou_thresh=args.iou)

    print("[info] Scripting wrapper (torch.jit.script)...")
    scripted = torch.jit.script(wrapper)
    scripted.eval()

    # Debug: check code for torchvision::nms presence
    try:
        code = scripted.code
        if "torchvision::nms" in code or "torchvision" in code:
            print("[warning] Scripted code contains reference to torchvision ops! Inspect scripted.code.")
        else:
            print("[info] Scripted code seems free of torchvision::nms (ok).")
    except Exception as e:
        print("[debug] Could not read scripted.code:", e)

    print("[info] Testing scripted wrapper on example input...")
    with torch.no_grad():
        test_out = scripted(example)
    print("[info] scripted wrapper output shape:", test_out.shape)

    # Optimize and save for lite interpreter
    print("[info] Optimizing for mobile...")
    optimized = optimize_for_mobile(scripted)
    print("[info] Saving to:", args.out_ptl)
    optimized._save_for_lite_interpreter(args.out_ptl)
    print("[success] Saved mobile .ptl:", args.out_ptl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", type=str, default=None, help="Path to trained .pt (optional, used to export via ultralytics)")
    parser.add_argument("--ts_path", type=str, default=None, help="Path to existing .torchscript (optional)")
    parser.add_argument("--out_ptl", type=str, default="gtsdb_yolo_416.ptl", help="Output .ptl path")
    parser.add_argument("--input_size", type=int, default=416, help="Model input size (H=W)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for postprocessing")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    args = parser.parse_args()
    main(args)
