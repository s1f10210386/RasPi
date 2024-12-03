import tensorflow as tf
import cv2
import numpy as np

def load_movenet_model():
    model = tf.saved_model.load('movenet_model')
    return model

def detect_poses(image_path, model):
    # 画像読み込み
    image = cv2.imread(image_path)
    input_image = tf.convert_to_tensor(image)
    input_image = tf.expand_dims(input_image, axis=0)
    
    # 姿勢推定
    results = model(input_image)
    
    return results

def main():
    model = load_movenet_model()
    image_path = 'test_image.jpg'
    poses = detect_poses(image_path, model)
    
    # 結果の表示・保存処理
    print(poses)

if __name__ == '__main__':
    main()