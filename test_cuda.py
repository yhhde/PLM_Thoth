import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Available:", torch.cuda.is_available())
