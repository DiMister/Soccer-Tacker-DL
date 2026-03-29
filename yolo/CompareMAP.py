from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(REPO_ROOT / "training" / "football.yaml")
MODEL_A = str(REPO_ROOT / "yolo26n3_100epoch_best.pt")
MODEL_B = str(REPO_ROOT / "fine_tuned_models" / "mosaic_tuned_yolo" / "weights" / "best.pt")
SPLIT = "val"
IMGSZ = 640
CONF = 0.001 # Lowered confidence threshold to capture more predictions for comparison
IOU = 0.6 # Increased IoU threshold to be more lenient in matching predictions to GT boxes, which can help show differences in mAP more clearly
BATCH = 16
DEVICE = None
WORKERS = 0
WINNER_METRIC = "map"


@dataclass
class EvalResult:
	name: str
	model_path: str
	map5095: float
	map50: float
	map75: float


def _try_get_attr(obj: Any, attr_path: str) -> Any:
	parts = attr_path.split(".")
	current = obj
	for part in parts:
		if current is None or not hasattr(current, part):
			return None
		current = getattr(current, part)
	return current


def _to_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _extract_metric(metrics: Any, attr_candidates: list[str], key_candidates: list[str]) -> float:
	for attr in attr_candidates:
		value = _to_float(_try_get_attr(metrics, attr))
		if value is not None:
			return value

	results_dict = getattr(metrics, "results_dict", None)
	if isinstance(results_dict, dict):
		for key in key_candidates:
			value = _to_float(results_dict.get(key))
			if value is not None:
				return value

	return 0.0


def evaluate_model(
	model_path: str,
	data_yaml: str,
	split: str,
	imgsz: int,
	conf: float,
	iou: float,
	batch: int,
	device: str | None,
	workers: int,
) -> EvalResult:
	model = YOLO(model_path)
	metrics = model.val(
		data=data_yaml,
		split=split,
		imgsz=imgsz,
		conf=conf,
		iou=iou,
		batch=batch,
		device=device,
		workers=workers,
		verbose=False,
		plots=False,
	)

	map5095 = _extract_metric(
		metrics,
		attr_candidates=["box.map"],
		key_candidates=["metrics/mAP50-95(B)", "metrics/mAP50-95"],
	)
	map50 = _extract_metric(
		metrics,
		attr_candidates=["box.map50"],
		key_candidates=["metrics/mAP50(B)", "metrics/mAP50"],
	)
	map75 = _extract_metric(
		metrics,
		attr_candidates=["box.map75"],
		key_candidates=["metrics/mAP75(B)", "metrics/mAP75"],
	)

	return EvalResult(
		name=Path(model_path).stem,
		model_path=model_path,
		map5095=map5095,
		map50=map50,
		map75=map75,
	)


def print_comparison(result_a: EvalResult, result_b: EvalResult, winner_metric: str) -> None:
	label_a = f"{result_a.name} ({result_a.model_path})"
	label_b = f"{result_b.name} ({result_b.model_path})"

	print("\nValidation mAP comparison")
	print("=" * 90)
	print(f"{'Model':<56} {'mAP50-95':>10} {'mAP50':>10} {'mAP75':>10}")
	print("-" * 90)
	print(
		f"{label_a:<56} {result_a.map5095:>10.4f} {result_a.map50:>10.4f} {result_a.map75:>10.4f}"
	)
	print(
		f"{label_b:<56} {result_b.map5095:>10.4f} {result_b.map50:>10.4f} {result_b.map75:>10.4f}"
	)
	print("=" * 90)

	metric_label = {
		"map": "mAP50-95",
		"map50": "mAP50",
		"map75": "mAP75",
	}[winner_metric]

	value_a = {
		"map": result_a.map5095,
		"map50": result_a.map50,
		"map75": result_a.map75,
	}[winner_metric]
	value_b = {
		"map": result_b.map5095,
		"map50": result_b.map50,
		"map75": result_b.map75,
	}[winner_metric]

	if abs(value_a - value_b) < 1e-12:
		print(f"Tie on {metric_label}: {value_a:.4f}")
		return

	winner = result_a if value_a > value_b else result_b
	margin = abs(value_a - value_b)
	print(f"Winner by {metric_label}: {winner.name} (+{margin:.4f})")


def main() -> int:
	data_yaml_path = Path(DATA_YAML)
	if not data_yaml_path.exists():
		raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

	print(f"Evaluating model A: {MODEL_A}")
	result_a = evaluate_model(
		model_path=MODEL_A,
		data_yaml=str(data_yaml_path),
		split=SPLIT,
		imgsz=IMGSZ,
		conf=CONF,
		iou=IOU,
		batch=BATCH,
		device=DEVICE,
		workers=WORKERS,
	)

	print(f"Evaluating model B: {MODEL_B}")
	result_b = evaluate_model(
		model_path=MODEL_B,
		data_yaml=str(data_yaml_path),
		split=SPLIT,
		imgsz=IMGSZ,
		conf=CONF,
		iou=IOU,
		batch=BATCH,
		device=DEVICE,
		workers=WORKERS,
	)

	print_comparison(result_a, result_b, WINNER_METRIC)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())