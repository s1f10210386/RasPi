# import cv2

# # カメラを起動（0はデフォルトカメラを指定）
# cap = cv2.VideoCapture(0)  # ラズパイのカメラモジュールでは VideoCapture(0) で動作する場合が多い

# if not cap.isOpened():
#     print("カメラが開けません。接続を確認してください。")
#     exit()

# # 1フレームをキャプチャ
# ret, frame = cap.read()

# if ret:
#     # フレームを保存
#     filename = "captured_image.jpg"
#     cv2.imwrite(filename, frame)
#     print(f"静止画を保存しました: {filename}")
# else:
#     print("フレームのキャプチャに失敗しました。")

# # カメラを解放
# cap.release()

import tensorflow as tf

# 解凍したモデルのパスを指定
model_path = "3.tflite"

# TensorFlow Lite Interpreterをロード
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("MoveNetモデルのロードが完了しました！")

# 入力と出力テンソルの確認
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("入力テンソル情報:", input_details)
print("出力テンソル情報:", output_details)
