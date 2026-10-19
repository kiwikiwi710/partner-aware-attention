# MT-HTA

這個專案使用 Multi-Task Hierarchical Task-Attention Transformer，根據桌球擊球歷史預測下一拍的擊球類型、正反手、落點、力道與旋轉。

目前支援原始桌球資料、擴充桌球資料，以及使用 pairID 前處理格式的雙打資料。

## 1. 第一次設定

請先安裝 Git、Miniconda/Anaconda。若要使用 GPU，還需要 NVIDIA 驅動程式。

```powershell
git clone https://github.com/kiwikiwi710/partber-aware-attention.git
cd partber-aware-attention
conda env create -f environment.yml
conda activate mt-hta
```

確認 Python 與 PyTorch：

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```

應使用 `mt-hta` 環境。`True` 表示 CUDA 可用；若是 `False`，仍可使用 CPU，但速度較慢。

所有指令都應從專案根目錄執行：

```powershell
cd "D:\path\to\partber-aware-attention"
conda activate mt-hta
```

## 2. 原始單打資料

GitHub 不包含前處理產生的 `.pkl` 與 `.json`，第一次使用時要重新產生：

```powershell
python preprocessing/TableTennis_preprocess.py --variant original
```

輸出位置：

```text
data/table_tennis/processed_data/
├── train_data.pkl
├── val_data.pkl
├── test_data.pkl
├── config.json
└── player_map.json
```

預設以 `match_id` 為單位切分：訓練集 70%、驗證集 10%、測試集 20%。

擴充桌球資料：

```powershell
python preprocessing/TableTennis_preprocess.py --variant all
```

## 3. 雙打資料

雙打 CSV 位於：

```text
data/table_tennis/double/dataset_doubles_all.csv
```

執行雙打前處理：

```powershell
cd data/table_tennis/double
python baseline_preprocess.py
cd ..\..\..
```

腳本會輸出到 `data/table_tennis/double/processed_baseline_data/`。目前模型預設讀取 `data/table_tennis/processed_data/`，所以要複製：

```powershell
Copy-Item `
  data\table_tennis\double\processed_baseline_data\* `
  data\table_tennis\processed_data\ `
  -Force
```

雙打前處理會覆蓋目前的 `processed_data`。若要切回原始單打：

```powershell
python preprocessing/TableTennis_preprocess.py --variant original
```

## 4. 先跑 1 epoch

```powershell
python scripts/run_all_models.py `
  --sport table_tennis `
  --models task_attention_itransformer_feature_token `
  --epochs 1 `
  --batch_size 128 `
  --seed 42 `
  --disable_mlflow
```

這會依序執行訓練、驗證與測試。

## 5. 正式訓練 40 epochs

原始單打與雙打使用相同模型指令；差別只在 `processed_data` 目前是哪一份資料。

```powershell
python scripts/run_all_models.py `
  --sport table_tennis `
  --models task_attention_itransformer_feature_token `
  --epochs 40 `
  --batch_size 128 `
  --seed 42 `
  --disable_mlflow `
  --results_base_dir outputs/results/table_tennis_40ep
```

結果會放在 `outputs/results/`，包括：

```text
train/weights/best_model.pth
train/weights/last_model.pth
test/evaluation_results.csv
results_summary.csv
```

## 6. 測試已訓練模型

```powershell
python scripts/test_location_loss.py `
  --sport table_tennis `
  --run_dir "outputs\results\table_tennis_40ep\RUN_DIRECTORY" `
  --disable_mlflow
```

將 `RUN_DIRECTORY` 換成實際的 run 資料夾名稱。

## 7. GitHub 同步

家裡電腦取得最新版本：

```powershell
cd partber-aware-attention
conda activate mt-hta
git pull
```

修改後上傳：

```powershell
git add 修改的檔案
git commit -m "Describe your changes"
git pull --rebase
git push
```

前處理產生的 `.pkl`、`.json`、模型權重與 `outputs/` 不會被 Git 追蹤；換電腦後需要重新執行前處理。

## 8. 常見問題

### 找不到 torch

```powershell
conda activate mt-hta
python -c "import torch; print(torch.__version__)"
```

### 找不到 scripts/run_all_models.py

請回到專案根目錄：

```powershell
cd "D:\path\to\partber-aware-attention"
```

### 終端中文亂碼

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

### results_summary.csv 被鎖定

關閉目前由 Excel、VS Code 或其他程式開啟的 `results_summary.csv`，再執行程式。
