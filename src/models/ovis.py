"""
Load và cấu hình model Ovis với quantization.
"""
import gc
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.utils import _get_env_path


MODEL_PATHS = {
    "ovis": _get_env_path("VQA_MODEL_OVIS", "AIDC-AI/Ovis2.5-9B"),
}


def get_model_path(model_key: str) -> str:
    """Trả về HuggingFace path tương ứng với model_key."""
    if model_key not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available keys: {list(MODEL_PATHS.keys())}"
        )
    return MODEL_PATHS[model_key]


def load_model(model_key: str = "ovis", bits: int | None = 8):
    """
    Load model với BitsAndBytes quantization.

    Args:
        model_key: Key trong MODEL_PATHS.
        bits: 4 hoặc 8 bit quantization.
    """
    torch.cuda.empty_cache()
    gc.collect()

    model_path = get_model_path(model_key)
    if bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    elif bits is None:
        quantization_config = None
    else:
        raise ValueError(f"bits phải là 4 or 8 or None, nhận được: {bits}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.eval()

    print(f"Đã load model: {model_path}")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        print(f"   GPU {i} Memory Used: {mem:.2f} GB")

    return model
