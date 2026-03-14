"""
Chạy inference trên toàn bộ test set.

Ví dụ:

"""
import argparse
import os
import sys
import json

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ovis import load_model
from src.dataset import VQADataset, custom_collate_fn
from src.inference import infer, compute_anls
from src.utils import set_seed

WANDB_LOG_CHUNK = 50
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,  help="Thư mục chứa data")
    parser.add_argument("--test_json",  required=True,  help="Đường dẫn tới file test JSON")
    parser.add_argument("--model_key",  default="ovis")
    parser.add_argument("--bits",       default=None, help="Số bit quantization (4, 8 hoặc None)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--use_wandb",       action="store_true",           help="Bật logging lên Weights & Biases")
    parser.add_argument("--wandb_project",   default="") 
    parser.add_argument("--wandb_run_name",  default=None,                  help="Tên run (mặc định: wandb tự đặt)")
    parser.add_argument("--wandb_api_key",   default=None)

    return parser.parse_args()


# =============================================
# WANDB HELPERS
# =============================================
 
def init_wandb(args):
    """Đăng nhập và khởi tạo wandb run."""
    import wandb
 
    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()  # Sẽ hỏi key nếu chưa đăng nhập
 
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_key":  args.model_key,
            "bits":       args.bits,
            "batch_size": args.batch_size,
            "test_json":  args.test_json,
            "dataset":    "ViInfographicVQA",
            "metric":     "ANLS",
        },
    )
    print(f"Wandb run: {run.url}")
    return run
 
 
def make_wandb_table():
    """Tạo wandb.Table với schema cố định."""
    import wandb
    return wandb.Table(columns=[
        "question_id", "image", "image_type", "answer_source",
        "element", "question", "ground_truth", "prediction",
        "anls", "exact_match",
    ])
 
 
def flush_wandb_table(table, step_anls, step_em, chunk_idx):
    """Log table lên wandb rồi trả về table mới (reset để tránh lỗi đóng băng)."""
    import wandb
    wandb.log({
        "eval/predictions": table,
        "eval/step_anls":   step_anls,
        "eval/step_em":     step_em,
        "eval/chunk":       chunk_idx,
    })
    return make_wandb_table()  # reset
 
 
# =============================================
# MAIN
# =============================================
 
def main():
    args = parse_args()
    set_seed(args.seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # --- Khởi tạo wandb (nếu bật) ---
    wandb_run = None
    wandb_table = None
    if args.use_wandb:
        wandb_run   = init_wandb(args)
        wandb_table = make_wandb_table()
 
    # --- Load model ---
    model = load_model(model_key=args.model_key, bits=args.bits)
 
    # --- Load dataset ---
    test_dataset = VQADataset(
        json_path=os.path.join(args.data_dir, args.test_json),
        data_dir=args.data_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )
 
    # --- Inference loop ---
    records    = []
    sum_anls   = 0.0
    total      = 0
    chunk_idx  = 0
 
    with torch.inference_mode():
        pbar = tqdm(test_loader, desc="Inference")
 
        for batch in pbar:
            for i in range(len(batch["question"])):
                img_path     = batch["image_path"][i]
                question     = batch["question"][i]
                gt           = batch["raw_answer"][i]
 
                pred  = infer(model, question, img_path)
                anls  = compute_anls(gt.lower(), pred)
                em    = int(gt.lower() == pred)
 
                # Lưu record CSV
                records.append({
                    "question_id":   batch["question_id"][i],
                    "image_type":    batch.get("image_type",    [""])[i],
                    "answer_source": batch.get("answer_source", [""])[i],
                    "element":       batch.get("element",       [""])[i],
                    "question":      question,
                    "ground_truth":  gt,
                    "prediction":    pred,
                    "anls":          anls,
                    "exact_match":   em,
                })
 
                # Thêm vào wandb Table
                if args.use_wandb:
                    import wandb
                    wandb_table.add_data(
                        batch["question_id"][i],
                        wandb.Image(img_path),          # Hiển thị ảnh trực tiếp trên wandb
                        batch.get("image_type",    [""])[i],
                        batch.get("answer_source", [""])[i],
                        batch.get("element",       [""])[i],
                        question,
                        gt,
                        pred,
                        anls,
                        em,
                    )
 
                sum_anls += anls
                total    += 1
 
            # Cập nhật progress bar
            cur_anls = sum_anls / total
            cur_em   = sum(r["exact_match"] for r in records) / total * 100
            pbar.set_postfix(ANLS=f"{cur_anls:.4f}", EM=f"{cur_em:.2f}%", n=total)
 
            # Log metrics mỗi bước lên wandb
            if args.use_wandb:
                import wandb
                wandb.log({
                    "eval/running_anls": cur_anls,
                    "eval/running_em":   cur_em,
                    "eval/total_samples": total,
                })
 
            # Flush table mỗi WANDB_LOG_CHUNK batch (tránh đơ)
            if args.use_wandb and total % WANDB_LOG_CHUNK == 0:
                wandb_table = flush_wandb_table(wandb_table, cur_anls, cur_em, chunk_idx)
                chunk_idx  += 1
 
    # --- Kết quả cuối ---
    avg_anls = sum_anls / total if total > 0 else 0
    avg_em   = sum(r["exact_match"] for r in records) / total * 100 if total > 0 else 0
    print(f"\nKết quả: ANLS = {avg_anls:.4f} | EM = {avg_em:.2f}%")
 
    # Flush phần còn lại của table + log summary lên wandb
    if args.use_wandb:
        import wandb
        if len(wandb_table.data) > 0:
            flush_wandb_table(wandb_table, avg_anls, avg_em, chunk_idx)
 
        # Summary hiện lên trang overview của run
        wandb.summary["final_anls"] = avg_anls
        wandb.summary["final_em"]   = avg_em
        wandb.summary["total_samples"] = total
        wandb_run.finish()
        print("📡 Đã đẩy kết quả lên Weights & Biases.")
 

 
 
if __name__ == "__main__":
    main()
 