Use the following command to check CUDA version on your GPU

```bash
nvidia-smi
```

# Resume training from a previous run

`TrainYOLO.py` now accepts an optional run name argument to resume from the last checkpoint.

Resume command format:

```bash
python training/TrainYOLO.py football_yolo26n{num}
```

Example:

```bash
python training/TrainYOLO.py football_yolo26n5
```

This looks for the checkpoint at:

```text
runs/train/football_yolo26n5/weights/last.pt
```

If the checkpoint file does not exist, the script will stop and print the missing path.

Might have to install different version of torch if DeviceDetect.py doesn't find your GPU

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Help training paramaters in "Advanced Training"

---

### 1. `mosaic=1.0` (The "Tiler")

- **What it does:** It takes 4 random training images and stitches them together into a single 2x2 grid (a "mosaic").
- **How it helps soccer detection:**
  - **Small Object Scaling:** Because it squeezes 4 images into the space of 1, the players become much smaller. This forces the model to get better at detecting "tiny" players (like those on the far side of the pitch).
  - **Context Variety:** It places players in unusual positions. A player might appear at the very edge of the frame or next to a piece of the stadium that wasn't in the original shot.
  - **Value 1.0:** This means there is a 100% chance that every image the model sees during training will be a mosaic.

### 2. `mixup=0.1` (The "Ghosting" Effect)

- **What it does:** it takes two different images and overlays them on top of each other with some transparency (like a double exposure in photography).
- **How it helps soccer detection:**
  - **Handling Occlusion:** In soccer, players are constantly running in front of each other (occlusion). Mixup creates "fake" occlusions where one player's silhouette is visible through another.
  - **Reducing Overconfidence:** It prevents the model from becoming too "sure" of itself based on simple backgrounds. By making the images "messy," the model learns to look for the specific textures of a player (the jersey, the legs) rather than just a shape against green grass.
  - **Value 0.1:** This means there is a 10% chance that an image will have this transparency effect applied. You don't want this too high (like 0.5), or the images become so "noisy" the model gets confused.

### 3. `perspective=0.0005` (The "Camera Angle" Simulator)

- **What it does:** it performs a 3D transformation on the image, tilting and warping it slightly to simulate a change in camera perspective.
- **How it helps soccer detection:**
  - **Camera Versatility:** Most soccer data comes from specific broadcast angles. However, if your model is used on a handheld camera or a different stadium height, the players' proportions change (e.g., they look shorter or wider).
  - **Distortion Correction:** It helps the model realize that a "player" is still a "player" even if the camera is tilted or if they are at the very corner of a wide-angle lens where stretching occurs.
  - **Value 0.0005:** This is a very subtle value. In soccer, you don't want to warp the pitch _too_ much (otherwise, the ground looks like a wall), but this small amount adds just enough spatial variety to make the model robust.

---

### Summary for your Football Project

| Parameter       | Goal       | Specific Soccer Benefit                                       |
| :-------------- | :--------- | :------------------------------------------------------------ |
| **Mosaic**      | Scale      | Helps detect players at the far end of the field.             |
| **Mixup**       | Robustness | Helps detect players in crowded penalty boxes.                |
| **Perspective** | Variety    | Helps the model work across different stadium camera heights. |

**Pro Tip:** If you notice your model is struggling to find the ball or players far away, **Mosaic** is your best friend. If your model is "missing" players when they stand in a group, **Mixup** is the key.
