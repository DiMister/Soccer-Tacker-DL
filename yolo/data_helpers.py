from pathlib import Path
from typing import List, Tuple


def list_val_folders(val_root: Path) -> List[Path]:
	return sorted([p for p in val_root.iterdir() if p.is_dir()])


def resolve_image_path(folder_path: Path, image_number: int) -> Path:
	img_dir = folder_path / "img1"
	if not img_dir.exists():
		raise FileNotFoundError(f"Missing img1 folder in {folder_path}")

	candidates = []
	for pad in (6, 8):
		stem = str(image_number).zfill(pad)
		for ext in (".jpg", ".jpeg", ".png"):
			candidates.append(img_dir / f"{stem}{ext}")

	for candidate in candidates:
		if candidate.exists():
			return candidate

	images = sorted([p for p in img_dir.iterdir() if p.is_file()])
	if not images:
		raise FileNotFoundError(f"No images found in {img_dir}")

	index = image_number - 1
	if 0 <= index < len(images):
		return images[index]

	raise FileNotFoundError(
		f"Image number {image_number} not found as file name or index in {img_dir}"
	)


def parse_gt_boxes(gt_path: Path, frame_id: int) -> List[Tuple[float, float, float, float]]:
	boxes = []
	if not gt_path.exists():
		return boxes

	with gt_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			parts = [p.strip() for p in line.split(",")]
			if len(parts) < 6:
				continue
			try:
				frame = int(float(parts[0]))
			except ValueError:
				continue
			if frame != frame_id:
				continue
			try:
				left = float(parts[2])
				top = float(parts[3])
				width = float(parts[4])
				height = float(parts[5])
			except ValueError:
				continue
			boxes.append((left, top, left + width, top + height))

	return boxes


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0.0, inter_x2 - inter_x1)
	inter_h = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area
	if union == 0:
		return 0.0
	return inter_area / union
