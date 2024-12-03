import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2

# MoveNet Lightningモデルのロード
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# テスト用画像のロード
image = cv2.imread("test_image.jpg")
input_image = cv2.resize(image, (192, 192))  # モデルの入力サイズにリサイズ
input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

# 推論実行
outputs = model(input_image)
keypoints = outputs["output_0"].numpy()

# 結果を描画
for keypoint in keypoints[0][0]:
    y, x, confidence = keypoint
    if confidence > 0.5:  # 信頼度が高い場合のみ描画
        cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)

# 結果を表示
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
