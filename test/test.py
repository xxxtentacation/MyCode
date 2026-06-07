import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"内置CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"GPU是否可用: {torch.cuda.is_available()}")
print(f"显卡名称: {torch.cuda.get_device_name(0)}")