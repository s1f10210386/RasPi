# from preprocess import preprocess_image
# from inference import load_model, run_inference
# import cv2

# # モデルと画像のパス
# model_path = "3.tflite"
# image_path = "test_image.jpg"

# # モデルをロード
# interpreter = load_model(model_path)

# # 入力画像を前処理
# input_image = preprocess_image(image_path)

# # 推論を実行
# output = run_inference(interpreter, input_image)

# # 推論結果を解析
# keypoints = output[0][0]  # [1, 1, 17, 3] の形状なので [0][0] でアクセス

# # 可視化用の元画像を読み込む
# original_image = cv2.imread(image_path)
# height, width, _ = original_image.shape

# # 各関節を描画
# for idx, keypoint in enumerate(keypoints):
#     x, y, confidence = keypoint
#     if confidence > 0.01:  # 信頼度が高い場合だけ描画
#         cx, cy = int(x * width), int(y * height)  # 元画像サイズにスケール
#         cv2.circle(original_image, (cx, cy), 5, (0, 255, 0), -1)  # 緑の点を描画

# # 結果を保存
# result_path = "result_image.jpg"
# cv2.imwrite(result_path, original_image)
# print(f"結果画像を保存しました: {result_path}")
# print("推論結果:", output)

# cx, cy = int(x * width), int(y * height)
# print(f"スケール後の座標: ({cx}, {cy})")

import tensorflow as tf
import numpy as np
import cv2

# モデルと画像のパス
model_path = "3.tflite"
image_path = "test_image.jpg"

# TensorFlowを使用した画像前処理
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_image = tf.image.resize_with_pad(image, 192, 192)  # パディング込みでリサイズ
    input_image = tf.cast(input_image, dtype=tf.float32) / 255.0  # 正規化
    input_image = tf.expand_dims(input_image, axis=0)  # バッチ次元を追加
    return input_image

# モデルのロード
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 推論の実行
def run_inference(interpreter, input_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# 前処理
input_image = preprocess_image(image_path)

# モデルロードと推論
interpreter = load_model(model_path)
keypoints = run_inference(interpreter, input_image)

# 結果の確認
print("推論結果:", keypoints)

# 可視化用の元画像を読み込む
original_image = cv2.imread(image_path)
height, width, _ = original_image.shape

# 各関節を描画
for idx, keypoint in enumerate(keypoints[0][0]):  # [1, 1, 17, 3] の形状なので [0][0] でアクセス
    x, y, confidence = keypoint
    if confidence > 0.01:  # 信頼度が高い場合だけ描画
        cx, cy = int(x * width), int(y * height)  # 元画像サイズにスケール
        cv2.circle(original_image, (cx, cy), 5, (0, 255, 0), -1)  # 緑の点を描画
        print(f"関節 {idx}: スケール後の座標: ({cx}, {cy}), 信頼度: {confidence:.2f}")

# 結果を保存
result_path = "result_image.jpg"
cv2.imwrite(result_path, original_image)
print(f"結果画像を保存しました: {result_path}")