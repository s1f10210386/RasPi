# import cv2
# import numpy as np

# # 画像ファイルのパス
# image_path = "test_image.jpg"

# # 1. 画像を読み込む
# image = cv2.imread(image_path)
# if image is None:
#     raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

# # 2. 画像を192x192にリサイズ
# input_image = cv2.resize(image, (192, 192))

# # 3. 正規化（0～255の値を0～1にスケール）
# input_image = input_image.astype(np.float32) / 255.0

# # 4. モデルが期待する形に変換（バッチ次元を追加）
# input_image = np.expand_dims(input_image, axis=0)  # [1, 192, 192, 3]

# print("前処理が完了しました！入力データの形:", input_image.shape)

# 前処理(画像のリサイズと正規化)を行うモジュール
import cv2
import numpy as np

def preprocess_image(image_path, target_size=(192, 192)):
    """
    画像を指定されたサイズにリサイズし、正規化して返す。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
    # 画像をリサイズ
    image = cv2.resize(image, target_size)
    # 正規化
    image = image.astype(np.float32) / 255.0
    # バッチ次元を追加
    image = np.expand_dims(image, axis=0)
    return image
