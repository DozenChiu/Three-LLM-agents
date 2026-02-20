import os
import subprocess

# 設定影片所在的資料夾路徑 (如果腳本放在同一層，保持 "./" 即可)
video_dir = "./" 

for filename in os.listdir(video_dir):
    # 只處理 .mp4 檔案
    if filename.endswith(".mp4"):
        # 取得檔名主體，例如 "0001"
        base_name = os.path.splitext(filename)[0] 
        
        # 定義對應的輸出資料夾路徑
        output_folder = os.path.join(video_dir, base_name)
        
        # 確保資料夾存在 (雖然你的截圖看起來已經建好了，但加這行比較保險)
        os.makedirs(output_folder, exist_ok=True)
        
        input_path = os.path.join(video_dir, filename)
        
        # 設定輸出檔名格式，例如存進 0001/0001_000.mp4, 0001/0001_01.mp4
        output_pattern = os.path.join(output_folder, f"{base_name}_%02d.mp4")
        
        # 構建 ffmpeg 指令
        # -c copy: 直接複製串流不重新編碼 (速度超快)
        # -segment_time 5: 每 5 秒切一段
        # -reset_timestamps 1: 將切出來的每段小影片時間軸歸零重置
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", "5",
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]
        
        print(f"正在切割影片: {filename} ...")
        # 執行指令，並隱藏 ffmpeg 預設的一大堆文字輸出
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("所有影片切割完成！")
