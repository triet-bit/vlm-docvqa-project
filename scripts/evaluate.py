"""
Tính lại metrics từ file predictions đã có sẵn.

Ví dụ:
    python scripts/evaluate.py --pred_file outputs/predictions/results.csv
"""
import argparse
import pandas as pd

from src.inference import compute_anls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True, help="File CSV chứa predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pred_file)

    # Tính lại ANLS nếu cần
    if "anls" not in df.columns:
        df["anls"] = df.apply(
            lambda row: compute_anls(str(row["ground_truth"]).lower(), str(row["prediction"])),
            axis=1,
        )

    avg_anls = df["anls"].mean()
    avg_em   = df["exact_match"].mean() * 100

    print(f"📊 Kết quả đánh giá trên {len(df)} mẫu:")
    print(f"   ANLS:        {avg_anls:.4f}")
    print(f"   Exact Match: {avg_em:.2f}%")

    # Phân tích theo image_type nếu có
    if "image_type" in df.columns:
        print("\n📂 Phân tích theo image_type:")
        print(df.groupby("image_type")["anls"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
