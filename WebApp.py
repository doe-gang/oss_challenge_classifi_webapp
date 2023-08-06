from flask import Flask, request, jsonify
from PIL import Image
import joblib
import json
import pandas as pd
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

@app.route("/")
def main_start():
    return "<p>INDEX PAGE</p>"

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/echo', methods=['POST']) #post echo api
def post_echo_call():
    param = request.get_json()
    return jsonify(param)
#curl json 명령어
#curl -X POST -H "Content-Type: application/json" -d "{\"TEXT1\" : \"test\"}" http://127.0.0.1:5000/echo
#curl -X POST http://localhost:5000/echo -H "Content-Type: application/json" -d '{"parameter": "value"}'

@app.route("/upload_image", methods=["POST"])
def chk():
    img = Image.open(request.files['file']) #json key 가 file로 설정되어야함 files image등으로 다른 키명으로 설정시 못읽어서 400에러발생
    width, height = img.size
    imgsize = {}
    imgsize['width'] = width
    imgsize['height'] = height
    
    return json.dumps(imgsize)
    #return jsonify({"width" : width, "heigh": height})
    #return "oddsK"
#curl -F 'file=@./images/test1.jpg' -X POST http://127.0.0.1:5000/upload_image


# score predict모델을 로드하고 예측 실행
loaded_model = joblib.load('reg_model.pkl')
# predict_score code
@app.route('/predict_score', methods=["GET"])
def predict_score():
    # param = request.get_json()
    param = request.args.get("hours")
    print('디버그 파라미터: ', param)

    lst_param = []
    lst_param.append(param)
    # 14를 df에서 np로 변환하고 predict 수행
    X_test = pd.DataFrame({'hours': lst_param}).to_numpy()
    y_pred = loaded_model.predict(X_test)
    y_pred
    print('디버그 y_pred: ', y_pred)

    result = {}
    result['score'] = y_pred[0][0]
    return json.dumps(result)

# cat_dog 모델 로드
new_model = tf.keras.models.load_model('cats_dogs.h5')
image_size = (180, 180)

@app.route('/predict_cat_dog', methods=["POST"])
def predict_cat_dog():
    img = Image.open(request.files['file'])
    _ = img.save("_temp.jpg")

    # test_loss, test_acc = new_model.evaluate(x,  y, verbose=2)
    img = keras.utils.load_img(
        "_temp.jpg", target_size=image_size
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = new_model.predict(img_array)
    score = float(predictions[0])
    print("디버그 score", score)
    res = "This image is " + \
        str(100 * round(1 - score, 2)) + "% cat and " + \
        str(100 * round(score, 2)) + "% dog."

    if score > 0.5:
        class_name = "dog"
    else:
        class_name = "cat"

    result = {}
    result['result'] = res
    result['class_name'] = class_name
    return json.dumps(result)
if __name__ == "__main__":
    app.run()
