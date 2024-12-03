from preprocess import preprocess_image
from inference import load_model, run_inference
import cv2

# モデルと画像のパス
model_path = "3.tflite"
image_path = "test_image.jpg"

# モデルをロード
interpreter = load_model(model_path)

# 入力画像を前処理
input_image = preprocess_image(image_path)

# 推論を実行
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

# 結果を保存
result_path = "result_image.jpg"
cv2.imwrite(result_path, original_image)
print(f"結果画像を保存しました: {result_path}")
