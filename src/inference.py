# モデルのロードと推論を行うモジュール
import tensorflow as tf

def load_model(model_path):
    """
    TensorFlow Liteモデルをロードしてインタープリタを返す。
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, input_image):
    """
    モデルに画像を入力し、推論結果を取得して返す。
    """
    # 入力と出力の詳細を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 入力データをセット
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # 推論を実行
    interpreter.invoke()

    # 出力データを取得
    return interpreter.get_tensor(output_details[0]['index'])
