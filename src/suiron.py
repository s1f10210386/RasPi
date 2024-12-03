import cv2
import numpy as np
from inference import load_model, run_inference
from preprocess import preprocess_image

# モデルと画像のパス
model_path = "3.tflite"
image_path = "test_image.jpg"

# モデルのロード
interpreter = load_model(model_path)

# 入力画像の前処理
input_image = preprocess_image(image_path)

# 推論実行
output = run_inference(interpreter, input_image)

# 推論結果を解析
keypoints = output[0][0]  # [1, 1, 17, 3] の形状なので [0][0] でアクセス

# 可視化用の元画像を読み込む
original_image = cv2.imread(image_path)
height, width, _ = original_image.shape

# 各関節を描画
for idx, keypoint in enumerate(keypoints):
    x, y, confidence = keypoint
    if confidence > 0.5:  # 信頼度が高い場合だけ描画
        cx, cy = int(x * width), int(y * height)  # 元画像サイズにスケール
        cv2.circle(original_image, (cx, cy), 5, (0, 255, 0), -1)  # 緑の点を描画

# 関節間を線で結ぶ
# 接続するペアの定義（例: [左肩, 右肩] のような組み合わせ）
connections = [
    (5, 6), (5, 7), (7, 9),  # 左肩、右肩、左肘、左手首
    (6, 8), (8, 10),         # 右肘、右手首
    (5, 11), (6, 12),        # 左腰、右腰
    (11, 13), (13, 15),      # 左膝、左足首
    (12, 14), (14, 16)       # 右膝、右足首
]

for connection in connections:
    start_idx, end_idx = connection
    x1, y1, c1 = keypoints[start_idx]
    x2, y2, c2 = keypoints[end_idx]
    if c1 > 0.5 and c2 > 0.5:  # 両方の信頼度が高い場合
        start_point = (int(x1 * width), int(y1 * height))
        end_point = (int(x2 * width), int(y2 * height))
        cv2.line(original_image, start_point, end_point, (255, 0, 0), 2)  # 青い線を描画

# 結果を保存
result_path = "result_image.jpg"
cv2.imwrite(result_path, original_image)
print(f"結果画像を保存しました: {result_path}")

# 結果を表示
cv2.imshow("Pose Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
