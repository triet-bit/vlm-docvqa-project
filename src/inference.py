"""
Hàm inference và tính metric ANLS cho model VQA.
"""
import torch
from PIL import Image
from Levenshtein import ratio as levenshtein_ratio

SYSTEM_PROMPT = (
    "Answer the following question based solely on the image content "
    "concisely with a single term."
)


def parse_answer(response: str) -> str:
    """Trích xuất phần trả lời từ output của model."""
    if "Answer:" in response:
        return response.split("Answer:")[-1].strip()
    return response.strip()


def clean_prediction(text: str) -> str:
    """Chuẩn hóa prediction theo cách repo ViInfographicVQA."""
    return (
        text.lower().strip()
        .rstrip(".")
        .replace('"', "")
        .rstrip(">").lstrip("<")
    )


def compute_anls(target: str, prediction: str, tau: float = 0.5) -> float:
    """
    Tính ANLS score (dùng Levenshtein.ratio như paper ViInfographicVQA).
    """
    target = target.lower().strip()
    prediction = clean_prediction(prediction)
    if not target or not prediction:
        return 1.0 if target == prediction else 0.0
    score = levenshtein_ratio(prediction, target)
    return score if score >= tau else 0.0


def infer(model, question: str, image_path: str) -> str:
    """
    Chạy inference cho một cặp (question, image).

    Returns:
        Câu trả lời đã được parse và lowercase.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": f"Question: {question}\nAnswer:"},
        ]},
    ]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.inference_mode():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=False,
            enable_thinking_budget=False,
            max_new_tokens=100,
            thinking_budget=0,
            eos_token_id=model.text_tokenizer.eos_token_id,
        )
        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_answer(response).lower()
