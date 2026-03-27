import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset_football"
YAML_PATH = REPO_ROOT / "training" / "football.yaml"
RUNS_TRAIN_ROOT = REPO_ROOT / "runs" / "train"


def convert_split(split: str) -> None:
	split_dir = DATASET_ROOT / split
	for seq_dir in sorted(split_dir.iterdir()):
		if not seq_dir.is_dir():
			continue

		img1_dir = seq_dir / "img1"
		gt_path = seq_dir / "gt" / "gt.txt"
		seqinfo_path = seq_dir / "seqinfo.ini"
		if not img1_dir.exists():
			continue

		img_w, img_h = 1280, 720
		if seqinfo_path.exists():
			for line in seqinfo_path.read_text().splitlines():
				if line.startswith("imWidth="):
					img_w = int(line.split("=")[1])
				elif line.startswith("imHeight="):
					img_h = int(line.split("=")[1])

		frame_boxes: dict[int, list] = {}
		if gt_path.exists():
			with gt_path.open("r", encoding="utf-8") as fh:
				for line in fh:
					parts = [p.strip() for p in line.strip().split(",")]
					if len(parts) < 7:
						continue
					try:
						frame = int(float(parts[0]))
						active = int(float(parts[6]))
						left = float(parts[2])
						top = float(parts[3])
						w = float(parts[4])
						h = float(parts[5])
					except ValueError:
						continue
					if active != 1:
						continue
					frame_boxes.setdefault(frame, []).append((left, top, w, h))

		for img_path in sorted(img1_dir.iterdir()):
			if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
				continue
			stem = img_path.stem
			frame_id = int(stem) if stem.isdigit() else None
			boxes = frame_boxes.get(frame_id, []) if frame_id is not None else []
			label_path = img1_dir / f"{stem}.txt"
			with label_path.open("w", encoding="utf-8") as lf:
				for left, top, w, h in boxes:
					cx = (left + w / 2) / img_w
					cy = (top + h / 2) / img_h
					nw = w / img_w
					nh = h / img_h
					lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

	print(f"Labels written for split: {split}")


def write_yaml() -> None:
	content = (
		f"path: {DATASET_ROOT.as_posix()}\n"
		"train: train\n"
		"val: val\n"
		"nc: 1\n"
		"names: ['player']\n"
	)
	YAML_PATH.write_text(content, encoding="utf-8")
	print(f"Dataset YAML written: {YAML_PATH}")


def resolve_resume_checkpoint(run_name: str) -> Path:
	checkpoint = RUNS_TRAIN_ROOT / run_name / "weights" / "last.pt"
	if not checkpoint.exists():
		raise FileNotFoundError(
			f"Could not find checkpoint for run '{run_name}' at: {checkpoint}"
		)
	return checkpoint


def main(resume_run_name: str | None = None) -> int:
	print("Converting MOT annotations to YOLO labels...")
	convert_split("train")
	convert_split("val")

	write_yaml()

	if resume_run_name:
		checkpoint = resolve_resume_checkpoint(resume_run_name)
		print(f"Resuming training from: {checkpoint}")
		model = YOLO(str(checkpoint))
		model.train(resume=True)
	else:
		model = YOLO("yolov8n.pt")  # Start from pretrained YOLOv8n weights
		model.train(
			data=str(YAML_PATH),
			epochs=100,
			imgsz=640, # could be 1280, but 640 is faster for testing
			batch=16,
			workers=6,
			project=str(REPO_ROOT / "runs" / "train"),
			name="football_yolo26n",
	
			# --- Advanced Parameters ---
			optimizer='MuSGD',     # New optimizer for YOLO26
			patience=20,           # Early stopping if no improvement for 20 epochs
			lr0=0.01,              # Initial learning rate
			lrf=0.01,              # Final learning rate fraction
			
			# --- Augmentations for Sports ---
			mosaic=1.0,            # Combine 4 images to help with small objects
			mixup=0.1,             # Help the model handle occlusions/crowds
			perspective=0.0005,    # Simulate different camera angles
			close_mosaic=10,       # Turn off mosaic for the last 10 epochs for fine-tuning
		)

	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Train YOLO model or resume from an existing run name."
	)
	parser.add_argument(
		"resume_run_name",
		nargs="?",
		help="Optional run name to resume, e.g. football_yolo26n5",
	)
	args = parser.parse_args()
	raise SystemExit(main(args.resume_run_name))