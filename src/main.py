import tensorflow as tf
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# MoveNetモデルをロード
MODEL_PATH = "movenet_lightning.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def estimate_pose(image):
    # 画像をリサイズ
    input_image = cv2.resize(image, (192, 192))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype(np.float32)

    # モデル推論
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

# カメラを起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = estimate_pose(frame)
    print("姿勢推定結果:", keypoints)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
