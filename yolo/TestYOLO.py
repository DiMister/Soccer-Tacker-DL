from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

from data_helpers import iou, list_val_folders, parse_gt_boxes, resolve_image_path


def run_detection(image_path: Path, model_path: str, conf: float):
	model = YOLO(model_path)
	results = model.predict(str(image_path), conf=conf, verbose=False)
	boxes = []
	if results:
		for box in results[0].boxes:
			xyxy = box.xyxy[0].tolist()
			boxes.append(
				{
					"xyxy": (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
					"conf": float(box.conf[0]) if box.conf is not None else None,
					"cls": int(box.cls[0]) if box.cls is not None else None,
				}
			)
	return boxes


def draw_boxes(
	image_path: Path,
	gt_boxes: List[Tuple[float, float, float, float]],
	pred_boxes: List[dict],
	output_path: Path,
	show_image: bool,
) -> None:
	image = Image.open(image_path).convert("RGB")
	draw = ImageDraw.Draw(image)
	try:
		font = ImageFont.load_default()
	except OSError:
		font = None

	for gt in gt_boxes:
		draw.rectangle(gt, outline="lime", width=2)
		if font:
			draw.text((gt[0], max(0, gt[1] - 12)), "GT", fill="lime", font=font)

	for pred in pred_boxes:
		x1, y1, x2, y2 = pred["xyxy"]
		label = "P"
		if pred.get("conf") is not None:
			label = f"P {pred['conf']:.2f}"
		draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
		if font:
			draw.text((x1, max(0, y1 - 12)), label, fill="red", font=font)

	image.save(output_path)
	if show_image:
		image.show()


def main() -> int:
	folder_number = 4
	image_number = 1
	#model_path = "yolov8n.pt"

	conf_threshold = 0.25
	output_path = Path("output_with_boxes.jpg")
	show_image = True

	repo_root = Path(__file__).resolve().parent.parent
	val_root = repo_root / "dataset_football" / "val"
	model_path = str(
		repo_root / "runs" / "train" / "football_yolo26n2" / "weights" / "best.pt"
	)
	if not val_root.exists():
		raise FileNotFoundError(f"Validation folder not found: {val_root}")

	folders = list_val_folders(val_root)
	if not folders:
		raise FileNotFoundError(f"No validation folders found in {val_root}")

	folder_index = folder_number - 1
	if folder_index < 0 or folder_index >= len(folders):
		available = "\n".join(
			[f"{idx + 1}: {folder.name}" for idx, folder in enumerate(folders)]
		)
		raise ValueError(
			f"Folder number out of range. Available folders:\n{available}"
		)

	folder_path = folders[folder_index]
	image_path = resolve_image_path(folder_path, image_number)

	stem = image_path.stem
	frame_id = int(stem) if stem.isdigit() else image_number
	gt_path = folder_path / "gt" / "gt.txt"
	gt_boxes = parse_gt_boxes(gt_path, frame_id)

	pred_boxes = run_detection(image_path, model_path, conf_threshold)

	print(f"Folder: {folder_path.name}")
	print(f"Image: {image_path.name}")
	print(f"Frame id: {frame_id}")
	print(f"Predictions: {len(pred_boxes)}")
	print(f"GT boxes: {len(gt_boxes)}")

	if gt_boxes:
		best_ious = []
		for gt in gt_boxes:
			best_iou = 0.0
			for pred in pred_boxes:
				best_iou = max(best_iou, iou(gt, pred["xyxy"]))
			best_ious.append(best_iou)
		mean_iou = sum(best_ious) / len(best_ious)
		print(f"Mean best IoU: {mean_iou:.4f}")
		for idx, (gt, score) in enumerate(zip(gt_boxes, best_ious), start=1):
			print(f"GT {idx}: {gt} best_iou={score:.4f}")

	if pred_boxes:
		for idx, pred in enumerate(pred_boxes, start=1):
			print(
				f"Pred {idx}: {pred['xyxy']} conf={pred['conf']} cls={pred['cls']}"
			)

	draw_boxes(image_path, gt_boxes, pred_boxes, output_path, show_image)
	print(f"Saved overlay image: {output_path.resolve()}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
