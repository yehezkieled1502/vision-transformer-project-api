import torch


DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
