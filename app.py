from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
app = Flask(__name__)
dic = ['DOG', 'HORSE', 'ELEPHANT', 'BUTTERFLY', 'HEN', 'CAT', 'COW', 'SHEEP', 'SPIDER', 'SQUIRREL']
model = load_model('model.h5')


def predict_label(img_path):
    BOYUT = 224
    resim = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resim = cv2.resize(resim, (BOYUT, BOYUT))
    resim = np.array(resim).reshape(-1, BOYUT, BOYUT, 3)
    p = model.predict(resim)
    return dic[np.argmax(p[0])], p.max()


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route('/submit', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        img = request.files['my_image']
        img_path = "static/" + img.filename
        dosya_adi=img.filename
        print(img.filename)
        uzanti=os.path.splitext(dosya_adi)[1].lower()
        err=0
        if uzanti == ".jpeg":
            err=1
        img.save(img_path)
        p,per = predict_label(img_path)
    return render_template("index.html", prediction=p, percentage=per, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)