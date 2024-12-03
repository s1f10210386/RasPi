import tensorflow as tf
import numpy as np
import cv2

# MoveNetモデルのダウンロード
model = tf.saved_model.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')

def load_image(path):
    # 画像の読み込み
    img = cv2.imread(path)
    # 画像のリサイズ
    img = cv2.resize(img, (192, 192))
    # 画像の正規化
    img = img.astype('float32') / 255.0
    # バッチ次元の追加
    input_img = np.expand_dims(img, axis=0)
    return input_img, img

def draw_keypoints(image, keypoints, confidence_threshold=0.3):
    # キーポイントの描画
    for kp in keypoints[0][0]:
        y, x, c = kp
        if c > confidence_threshold:
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 4, (0, 255, 0), -1)

def main():
    input_image, display_image = load_image('../test_image.jpg')
    # 推論の実行
    outputs = model.signatures['serving_default'](tf.constant(input_image))
    keypoints = outputs['output_0'].numpy()
    # キーポイントの描画
    draw_keypoints(display_image, keypoints)
    # 結果の表示
    cv2.imshow('Pose Estimation', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
