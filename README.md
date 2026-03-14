# VLM-VQA Project

Đánh giá Vision Language Models trên dataset **ViInfographicVQA** (tiếng Việt).

## Cấu trúc thư mục

```
vlm-vqa-project/
├── data/                    # Dữ liệu (raw, processed, images)
├── src/                     # Source code chính
│   ├── dataset.py           # VQADataset, DataLoader utils
│   ├── inference.py         # infer(), compute_anls(), parse_answer()
│   ├── utils.py             # set_seed(), helper functions
│   └── models/
│       └── ovis.py          # Load model Ovis + quantization
├── configs/
│   ├── model_config.yaml    # Cấu hình model
│   └── infer_config.yaml    # Cấu hình inference
├── notebooks/               # Jupyter notebooks thử nghiệm
├── scripts/
│   ├── run_inference.py     # Chạy inference toàn bộ test set
│   └── evaluate.py          # Tính metrics từ file predictions
└── outputs/                 # Kết quả dự đoán và logs
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

**Chạy inference:**
```bash
python scripts/run_inference.py \
    --data_dir /path/to/ViInfographicVQA \
    --test_json data/single_test.json \
    --output outputs/predictions/results.csv \
    --bits 8
```

**Đánh giá kết quả:**
```bash
python scripts/evaluate.py --pred_file outputs/predictions/results.csv
```

## Model hỗ trợ

| Key    | Model                  |
|--------|------------------------|
| `ovis` | AIDC-AI/Ovis2.5-9B     |
