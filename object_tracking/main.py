from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------
# Editable tracking configuration
# ------------------------------
MODEL_PATH = Path("yolo26n3_100epoch_best.pt")
VAL_DIR = Path("dataset_football") / "val"
OUTPUT_DIR = Path("runs") / "tracking" / "botsort_val"

# Keep this as botsort.yaml to use BoT-SORT.
# Options: "botsort.yaml", "bytetrack.yaml"
TRACKER_CFG = "botsort.yaml"

CONF = 0.25
IOU = 0.5
IMGSZ = 1280
DEVICE: Optional[str] = None  # e.g. "cpu", "0"

SAVE_VIDEO = True
SEQUENCE: Optional[str] = "v_ITo3sCnpw_k_c007"  # e.g. "v_2QhNRucNC7E_c017"


def to_numpy(data: object) -> np.ndarray:
	if hasattr(data, "cpu"):
		data = data.cpu()  # type: ignore[assignment]
	if hasattr(data, "numpy"):
		data = data.numpy()  # type: ignore[assignment]
	return np.asarray(data)


def find_sequences(val_dir: Path, sequence_name: Optional[str]) -> Iterable[Path]:
	if sequence_name:
		seq_path = val_dir / sequence_name
		if not seq_path.exists():
			raise FileNotFoundError(f"Sequence not found: {seq_path}")
		return [seq_path]

	return sorted(p for p in val_dir.iterdir() if p.is_dir())


def load_seqinfo(seq_path: Path) -> tuple[int, int, float]:
	seqinfo_path = seq_path / "seqinfo.ini"
	width = 1280
	height = 720
	fps = 25.0

	if not seqinfo_path.exists():
		return width, height, fps

	for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
		if line.startswith("imWidth="):
			width = int(line.split("=", 1)[1])
		elif line.startswith("imHeight="):
			height = int(line.split("=", 1)[1])
		elif line.startswith("frameRate="):
			fps = float(line.split("=", 1)[1])

	return width, height, fps


def run_sequence(
	model: YOLO,
	seq_path: Path,
	output_dir: Path,
	tracker: str,
	conf: float,
	iou: float,
	imgsz: int,
	device: Optional[str],
	save_video: bool,
) -> None:
	img_dir = seq_path / "img1"
	if not img_dir.exists():
		print(f"[WARN] Missing image directory for {seq_path.name}: {img_dir}")
		return

	sequence_output_dir = output_dir / seq_path.name
	sequence_output_dir.mkdir(parents=True, exist_ok=True)

	labels_dir = output_dir / "labels"
	labels_dir.mkdir(parents=True, exist_ok=True)
	mot_txt = labels_dir / f"{seq_path.name}.txt"

	width, height, fps = load_seqinfo(seq_path)
	writer = None
	if save_video:
		video_path = sequence_output_dir / "tracked.mp4"
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
		writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

	lines: list[str] = []
	stream = model.track(
		source=str(img_dir),
		stream=True,
		tracker=tracker,
		persist=True,
		conf=conf,
		iou=iou,
		imgsz=imgsz,
		device=device,
		verbose=False,
	)

	for frame_idx, result in enumerate(stream, start=1):
		boxes = result.boxes

		if boxes is not None and boxes.id is not None and len(boxes) > 0:
			ids = to_numpy(boxes.id).astype(int).tolist()
			xyxy = to_numpy(boxes.xyxy).tolist()
			scores = to_numpy(boxes.conf).tolist()

			for track_id, box, score in zip(ids, xyxy, scores):
				x1, y1, x2, y2 = box
				w = max(0.0, x2 - x1)
				h = max(0.0, y2 - y1)
				lines.append(
					f"{frame_idx},{track_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{score:.6f},-1,-1,-1"
				)

		if writer is not None:
			plotted = result.plot()
			if plotted.shape[1] != width or plotted.shape[0] != height:
				plotted = cv2.resize(plotted, (width, height), interpolation=cv2.INTER_LINEAR)
			writer.write(plotted)

	mot_txt.write_text("\n".join(lines), encoding="utf-8")

	if writer is not None:
		writer.release()

	print(f"[DONE] {seq_path.name}: {len(lines)} tracks saved to {mot_txt}")


def main() -> None:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
	if not VAL_DIR.exists():
		raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(MODEL_PATH))
	sequences = find_sequences(VAL_DIR, SEQUENCE)

	for seq_path in sequences:
		run_sequence(
			model=model,
			seq_path=seq_path,
			output_dir=OUTPUT_DIR,
			tracker=TRACKER_CFG,
			conf=CONF,
			iou=IOU,
			imgsz=IMGSZ,
			device=DEVICE,
			save_video=SAVE_VIDEO,
		)

	print(f"All tracking results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
