"""
Run YOLO on frames from a SportsMOT soccer sequence and save annotated images.

What this script does:
- Loads a YOLO model (default: yolov8n.pt)
- Runs detection on frames in a sequence's img1/ folder
- Filters detections to the COCO "person" class (cls=0)
- Saves each annotated frame to an output folder 

Usage (example):
1) Use a sequence name (must exist under dataset_football/val/):
    python Scripts/run_yolo_on_sequence.py --seq_name "v_2QhNRucNC7E_c017" --out_dir "outputs/v_2QhNRucNC7E_c017" --step 10 --conf 0.35

Notes:
- --step x means "run on every xth frame".
- You can change conf to reduce false positives.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def draw_boxes(img, boxes, color=(0, 255, 0)):
    """Draws xyxy boxes on an image (OpenCV BGR)."""
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img,
            f"{conf:.2f}",
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return img


def resolve_seq_path(repo_root: Path, seq_path_arg: str | None, seq_name: str | None) -> Path:
    """
    Resolves which img1 folder to use.

    Priority:
    1) If --seq_path is provided: use it (absolute or relative).
    2) Else if --seq_name is provided: build path under repo_root/dataset_football/val/<seq_name>/img1

    This allows teammates to keep dataset under the repo folder (ignored by git),
    without hardcoding machine-specific absolute paths.
    """
    if seq_path_arg:
        # Allow absolute paths or paths relative to repo root
        p = Path(seq_path_arg)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    if seq_name:
        return (repo_root / "dataset_football" / "val" / seq_name / "img1").resolve()

    raise ValueError("You must provide either --seq_path or --seq_name.")

def with_step_suffix(out_dir: Path, step: int) -> Path:
    """
    Appends a step suffix to the output directory name.
    Example:
      outputs/v_abc_c017   + step=10  -> outputs/v_abc_c017_step10
      outputs              + step=30  -> outputs_step30
    """
    return out_dir.with_name(f"{out_dir.name}_step{step}")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    # Either provide a direct path to img1...
    group.add_argument(
        "--seq_path",
        type=str,
        help="Path to a sequence img1 folder (absolute or relative to repo root).",
    )

    # ...or just provide the sequence folder name under dataset_football/val/
    group.add_argument(
        "--seq_name",
        type=str,
        help='Sequence folder name under "dataset_football/val/". Example: v_2QhNRucNC7E_c017',
    )

    parser.add_argument("--out_dir", type=str, default="outputs", help="Where to save annotated images")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--step", type=int, default=30, help="Process every Nth frame")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # Scripts/ -> repo root
    seq_path = resolve_seq_path(repo_root, args.seq_path, args.seq_name)

    if not seq_path.exists():
        raise FileNotFoundError(f"img1 folder not found: {seq_path}")

    base_out_dir = (repo_root / args.out_dir).resolve()
    out_dir = with_step_suffix(base_out_dir, args.step)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    PERSON_CLASS = [0]  # COCO class 0 = person

    frames = sorted(seq_path.glob("*.jpg"))
    if not frames:
        # If your dataset uses PNG frames, include them too
        frames = sorted(seq_path.glob("*.png"))

    if not frames:
        raise FileNotFoundError(f"No frames found in: {seq_path} (expected .jpg or .png)")

    print(f"Using img1: {seq_path}")
    print(f"Found {len(frames)} frames. Processing every {args.step} frame(s)...")
    print(f"Saving outputs to: {out_dir}")

    for i, frame_path in enumerate(frames[:: args.step]):
        results = model.predict(
            str(frame_path),
            conf=args.conf,
            classes=PERSON_CLASS,
            verbose=False,
        )

        img = cv2.imread(str(frame_path))
        boxes = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            boxes.append((x1, y1, x2, y2, conf))

        img = draw_boxes(img, boxes)
        out_path = out_dir / frame_path.name
        cv2.imwrite(str(out_path), img)

        if i % 10 == 0:
            print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()