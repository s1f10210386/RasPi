import cv2

# カメラを起動（0はデフォルトカメラを指定）
cap = cv2.VideoCapture(0)  # ラズパイのカメラモジュールでは VideoCapture(0) で動作する場合が多い

if not cap.isOpened():
    print("カメラが開けません。接続を確認してください。")
    exit()

# 1フレームをキャプチャ
ret, frame = cap.read()

if ret:
    # フレームを保存
    filename = "captured_image.jpg"
    cv2.imwrite(filename, frame)
    print(f"静止画を保存しました: {filename}")
else:
    print("フレームのキャプチャに失敗しました。")

# カメラを解放
cap.release()
