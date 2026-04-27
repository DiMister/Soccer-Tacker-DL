Might have to install different version of torch if DeviceDetect.py doesn't find your GPU

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
