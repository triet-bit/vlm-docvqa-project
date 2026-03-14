"""
Các hàm tiện ích dùng chung toàn project.
"""
import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """Fix random seed để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_env_path(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """Lấy đường dẫn từ biến môi trường, trả về default nếu không tìm thấy."""
    value = os.environ.get(env_var)
    if value:
        return value
    return default


def extract_clean_model_name(model_path: str) -> str:
    """Trích xuất tên model sạch từ HuggingFace path."""
    name = model_path.split("/")[-1]
    if "_" in name:
        parts = name.split("_", 1)
        if len(parts) > 1:
            return parts[1]
    return name
