## 辨識類別

`glass`（玻璃）、`metal`（金屬）、`paper`（紙類）、`plastic`（塑膠）、`trash`（其他垃圾）

---

## 環境需求

- Python 3.13
- CUDA 12.8
- PyTorch 2.11.0+cu128 / Torchvision 0.26.0+cu128
- OpenCV 4.13.0

**安裝步驟：**

```bash
# 1. 安裝 PyTorch（需指定 CUDA 版本）
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. 安裝其他套件
pip install -r requirements.txt
```

---

## 使用流程

```bash
python prepare_dataset.py  # 整合原始資料集 → data/processed/
python split_dataset.py    # 切割資料集 → data/split/
python train.py            # 訓練模型
python evaluate.py         # 測試集評估
python demo.py             # 即時辨識（按Q結束）
```

---

## 資料集來源

本專案使用以下三個公開資料集，感謝原作者的貢獻：

| 資料集 | 來源 |
|---|---|
| **TrashNet** | Gary Thung & Mindy Yang, Stanford University |
| **Garbage Classification** | Kaggle — Garbage Classification Dataset |

各資料集依其原始授權條款使用，僅供學術研究與課程專題用途。
