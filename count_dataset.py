# count_dataset.py
# counts sequences, images, labels, and total labeled boxes in dataset_football

from pathlib import Path

DATASET_ROOT = Path("dataset_football")


def count_split(split_path: Path) -> dict:
    sequence_dirs = [p for p in split_path.iterdir() if p.is_dir()]

    image_count = 0
    label_count = 0
    box_count = 0

    for seq in sequence_dirs:
        img_dir = seq / "img1"

        # count images and label txt files inside img1
        if img_dir.exists():
            for file in img_dir.iterdir():
                if file.is_file():
                    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        image_count += 1
                    elif file.suffix.lower() == ".txt":
                        label_count += 1

                        # count labeled boxes by counting non-empty lines
                        try:
                            with open(file, "r", encoding="utf-8") as f:
                                lines = [line.strip() for line in f if line.strip()]
                                box_count += len(lines)
                        except Exception as e:
                            print(f"could not read {file}: {e}")

    return {
        "sequences": len(sequence_dirs),
        "images": image_count,
        "labels": label_count,
        "boxes": box_count,
    }


def main():
    for split_name in ["train", "val"]:
        split_path = DATASET_ROOT / split_name
        if split_path.exists():
            stats = count_split(split_path)
            print(f"\n{split_name.upper()} SPLIT")
            print(f"Sequences: {stats['sequences']}")
            print(f"Images:    {stats['images']}")
            print(f"Labels:    {stats['labels']}")
            print(f"Boxes:     {stats['boxes']}")
        else:
            print(f"\n{split_name.upper()} SPLIT")
            print("not found")


if __name__ == "__main__":
    main()