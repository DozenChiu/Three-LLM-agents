# Molmo2-4B-SFT 專案說明文件

## 1. 建立與設定環境

請依照以下步驟重新建立乾淨的執行環境：

1. **離開目前環境**
   ```bash
   conda deactivate
   ```

2. **刪除損壞的環境**(這個步驟可以跳過)
   ```bash
   conda env remove -n molmo2_cu124_backup
   ```

3. **重新建立基礎環境**
   ```bash
   conda env create -f environment.yml
   ```

4. **啟用新環境**
   ```bash
   conda activate molmo2_cu124_backup
   ```

5. **分批補回核心與衝突套件**
   ```bash
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
   pip install datasets meteor pyarrow
   pip install git+[https://github.com/allenai/molmo.git](https://github.com/allenai/molmo.git)
   ```

---

## 2. 專案目錄與架構說明

專案主要包含三個核心資料夾：`Molmo2`、`molmo_data` 與 `models/molmo2-4b`。

### 📁 核心模型與資料集
* **`molmo_data/`**：蝦隻資料集目錄。（因檔案過大，並未上傳至 GitHub）
* **`models/molmo2-4b/`**：專案計畫使用的 Baseline model 存放位置。因檔案過大，並未上傳至 GitHub）
* **`environment.yml`**：Conda 環境設定檔。

### 📁 核心程式碼 (Molmo2/)
此資料夾包含 Supervised Fine-Tuning (SFT) 以及模型評分測試的相關程式碼。

#### 訓練與權重 (Training & Checkpoints)
* **`train_shrimp_lora_hf.py`**：SFT 訓練腳本。建好環境後，於終端機執行 `python train_shrimp_lora_hf.py`。
* **`molmo_runs/shrimp_lora_full/`**：存放 SFT 訓練完成後的模型權重與結果。

#### 評估與測試 (Evaluation)
* **`eval_base_2_current_folder.sh`**：測試 Baseline model（未 Fine-tune）的自動化腳本。於終端機直接執行 `bash eval_base_2_current_folder.sh`。
* **`eval_current_folder_2.sh`**：測試 SFT model（已 Fine-tune）的自動化腳本。於終端機直接執行 `bash eval_current_folder_2.sh`。
* **`eval_molmo2_base_metrics_2`**：用於計算 Baseline model 評估分數的程式碼。
* **`eval_shrimp_metrics`**：用於計算 SFT model 評估分數的程式碼。

#### 應用展示 (Demo UI)
* **`gradio_shrimp_chat_2_betterUI.py`**：展示網站 (Demo Web UI) 的啟動程式。

### 📁 資料統計與分析 (shrimp/)
此資料夾主要用於計算與分析文本數量。

* **`count_shrimp_report_table.py`**：資料統計腳本。進入該資料夾後，於終端機執行 `python count_shrimp_report_table.py`。
    * **輸出結果**：統計畫面會直接顯示於終端機，亦可開啟生成的 `shrimp_dataset_summary.json` 查看詳細的統計結果。