# Molmo2-4B-SFT
### 1. 建立環境
# 1. 先離開目前環境
```conda deactivate```

# 2. 刪除這個不小心中毒的環境
```conda env remove -n molmo2_cu124_backup```

# 3. 重新建立基礎環境
```conda env create -f environment.yml```

# 4. 啟用新環境
```conda activate molmo2_cu124_backup```

# 5. 分批補回核心與衝突套件
```
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install datasets meteor pyarrow
pip install git+https://github.com/allenai/molmo.git
```