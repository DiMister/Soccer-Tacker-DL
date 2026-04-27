from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from ultralytics import YOLO

from scipy.optimize import linear_sum_assignment

FrameDetections = Dict[int, List[Tuple[int, np.ndarray]]]
PairCounts = Dict[Tuple[str, int, int], int]


# ==========================================
# Editable configuration for HOTA evaluation
# ==========================================
MODEL_PATH = Path("fine_tuned_models/mosaic_tuned_yolo/weights/best.pt")
VAL_DIR = Path("dataset_football") / "val"
OUTPUT_DIR = Path("runs") / "tracking" / "hota_eval"
TRACKER = "botsort.yaml"  # or "bytetrack.yaml"

CONF = 0.25
IOU = 0.5
IMGSZ = 1280
DEVICE: str | None = None  # e.g. "cpu", "0"

SEQUENCE: str | None = None  # e.g. "v_2QhNRucNC7E_c017" for single sequence


def to_numpy(data: object) -> np.ndarray:
	if hasattr(data, "cpu"):
		data = data.cpu()  # type: ignore[assignment]
	if hasattr(data, "numpy"):
		data = data.numpy()  # type: ignore[assignment]
	return np.asarray(data)


def iter_sequences(val_dir: Path, sequence_name: str | None) -> Iterable[Path]:
	if sequence_name:
		candidate = val_dir / sequence_name
		if not candidate.exists():
			raise FileNotFoundError(f"Sequence not found: {candidate}")
		return [candidate]
	return sorted(p for p in val_dir.iterdir() if p.is_dir())


def parse_mot_txt(txt_path: Path) -> FrameDetections:
	frames: FrameDetections = defaultdict(list)
	if not txt_path.exists():
		return frames

	for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line:
			continue
		parts = [p.strip() for p in line.split(",")]
		if len(parts) < 6:
			continue

		frame_id = int(float(parts[0]))
		track_id = int(float(parts[1]))
		x = float(parts[2])
		y = float(parts[3])
		w = float(parts[4])
		h = float(parts[5])

		if len(parts) >= 7:
			conf = float(parts[6])
			if conf <= 0:
				continue

		frames[frame_id].append((track_id, np.array([x, y, w, h], dtype=np.float64)))

	return frames


def iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
	ax1, ay1, aw, ah = box_a
	bx1, by1, bw, bh = box_b
	ax2, ay2 = ax1 + aw, ay1 + ah
	bx2, by2 = bx1 + bw, by1 + bh

	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)

	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	if inter <= 0:
		return 0.0

	union = aw * ah + bw * bh - inter
	if union <= 0:
		return 0.0
	return float(inter / union)


def greedy_match(iou_matrix: np.ndarray, alpha: float) -> List[Tuple[int, int]]:
	matches: List[Tuple[int, int]] = []
	used_g = set()
	used_p = set()

	candidates: List[Tuple[float, int, int]] = []
	for g in range(iou_matrix.shape[0]):
		for p in range(iou_matrix.shape[1]):
			score = float(iou_matrix[g, p])
			if score >= alpha:
				candidates.append((score, g, p))

	candidates.sort(reverse=True, key=lambda x: x[0])
	for _, g, p in candidates:
		if g in used_g or p in used_p:
			continue
		used_g.add(g)
		used_p.add(p)
		matches.append((g, p))

	return matches


def match_frame(gt_boxes: List[np.ndarray], pr_boxes: List[np.ndarray], alpha: float) -> List[Tuple[int, int]]:
	if not gt_boxes or not pr_boxes:
		return []

	iou_matrix = np.zeros((len(gt_boxes), len(pr_boxes)), dtype=np.float64)
	for g, gt_box in enumerate(gt_boxes):
		for p, pr_box in enumerate(pr_boxes):
			iou_matrix[g, p] = iou_xywh(gt_box, pr_box)

	if linear_sum_assignment is None:
		return greedy_match(iou_matrix, alpha)

	cost = 1.0 - iou_matrix
	large_cost = 1e6
	cost[iou_matrix < alpha] = large_cost
	r_idx, c_idx = linear_sum_assignment(cost)

	matches: List[Tuple[int, int]] = []
	for r, c in zip(r_idx.tolist(), c_idx.tolist()):
		if iou_matrix[r, c] >= alpha:
			matches.append((r, c))
	return matches


def run_tracking_and_save(
	model: YOLO,
	seq_path: Path,
	tracker: str,
	conf: float,
	iou: float,
	imgsz: int,
	device: str | None,
	pred_dir: Path,
) -> Path:
	img_dir = seq_path / "img1"
	if not img_dir.exists():
		raise FileNotFoundError(f"Missing image directory: {img_dir}")

	pred_dir.mkdir(parents=True, exist_ok=True)
	pred_file = pred_dir / f"{seq_path.name}.txt"

	lines: List[str] = []
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
		if boxes is None or boxes.id is None or len(boxes) == 0:
			continue

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

	pred_file.write_text("\n".join(lines), encoding="utf-8")
	print(f"[TRACK] {seq_path.name}: wrote {len(lines)} detections to {pred_file}")
	return pred_file


def compute_hota(
	gt_by_seq: Dict[str, FrameDetections], pred_by_seq: Dict[str, FrameDetections]
) -> Dict[str, float]:
	alphas = [a / 100 for a in range(5, 100, 5)]
	hota_values: List[float] = []
	deta_values: List[float] = []
	assa_values: List[float] = []

	for alpha in alphas:
		tp = 0
		fp = 0
		fn = 0
		pair_tp: PairCounts = defaultdict(int)
		gt_match_count: Dict[Tuple[str, int], int] = defaultdict(int)
		pr_match_count: Dict[Tuple[str, int], int] = defaultdict(int)

		for seq_name, gt_frames in gt_by_seq.items():
			pred_frames = pred_by_seq.get(seq_name, {})
			all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

			for frame_id in all_frames:
				gt_items = gt_frames.get(frame_id, [])
				pr_items = pred_frames.get(frame_id, [])

				gt_ids = [gid for gid, _ in gt_items]
				pr_ids = [pid for pid, _ in pr_items]
				gt_boxes = [box for _, box in gt_items]
				pr_boxes = [box for _, box in pr_items]

				matches = match_frame(gt_boxes, pr_boxes, alpha)
				tp += len(matches)
				fp += len(pr_items) - len(matches)
				fn += len(gt_items) - len(matches)

				for g_idx, p_idx in matches:
					g_key = (seq_name, gt_ids[g_idx])
					p_key = (seq_name, pr_ids[p_idx])
					pair_tp[(seq_name, gt_ids[g_idx], pr_ids[p_idx])] += 1
					gt_match_count[g_key] += 1
					pr_match_count[p_key] += 1

		deta = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

		if tp == 0:
			assa = 0.0
		else:
			assoc_sum = 0.0
			for (seq_name, gt_id, pr_id), pair_count in pair_tp.items():
				g_key = (seq_name, gt_id)
				p_key = (seq_name, pr_id)
				fpa = pr_match_count[p_key] - pair_count
				fna = gt_match_count[g_key] - pair_count
				den = pair_count + fpa + fna
				ass_iou = (pair_count / den) if den > 0 else 0.0
				assoc_sum += pair_count * ass_iou
			assa = assoc_sum / tp

		hota_alpha = math.sqrt(max(0.0, deta * assa))
		hota_values.append(hota_alpha)
		deta_values.append(deta)
		assa_values.append(assa)

	return {
		"HOTA": float(np.mean(hota_values)) if hota_values else 0.0,
		"DetA": float(np.mean(deta_values)) if deta_values else 0.0,
		"AssA": float(np.mean(assa_values)) if assa_values else 0.0,
	}


def main() -> None:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
	if not VAL_DIR.exists():
		raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

	model = YOLO(str(MODEL_PATH))
	pred_root = OUTPUT_DIR / MODEL_PATH.stem / "labels"
	sequences = list(iter_sequences(VAL_DIR, SEQUENCE))

	gt_by_seq: Dict[str, FrameDetections] = {}
	pred_by_seq: Dict[str, FrameDetections] = {}

	for seq_path in sequences:
		gt_file = seq_path / "gt" / "gt.txt"
		if not gt_file.exists():
			print(f"[WARN] Skipping {seq_path.name}, gt file missing: {gt_file}")
			continue

		pred_file = run_tracking_and_save(
			model=model,
			seq_path=seq_path,
			tracker=TRACKER,
			conf=CONF,
			iou=IOU,
			imgsz=IMGSZ,
			device=DEVICE,
			pred_dir=pred_root,
		)

		gt_by_seq[seq_path.name] = parse_mot_txt(gt_file)
		pred_by_seq[seq_path.name] = parse_mot_txt(pred_file)

	if not gt_by_seq:
		raise RuntimeError("No valid sequences were evaluated.")

	metrics = compute_hota(gt_by_seq, pred_by_seq)

	print("\n=== HOTA Evaluation (Validation Set) ===")
	print(f"Model: {MODEL_PATH}")
	print(f"Tracker: {TRACKER}")
	print(f"Sequences evaluated: {len(gt_by_seq)}")
	print(f"HOTA: {metrics['HOTA']:.4f}")
	print(f"DetA: {metrics['DetA']:.4f}")
	print(f"AssA: {metrics['AssA']:.4f}")
	print(f"Prediction files: {pred_root}")


if __name__ == "__main__":
	main()
