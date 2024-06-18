from flask import Flask, render_template, request
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'project\\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
g_model = tf.keras.models.load_model('project/g_model')


def resize_image(gray_img,target_size=(256, 256)):
    gray_img = gray_img.resize(target_size)
    gray_img_array = np.array(gray_img).astype('float32')
    gray_img_array = (gray_img_array - 127.5) / 127.5
    gray_img_array = gray_img_array.reshape(1, 256, 256, 1)
    return gray_img_array


@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'POST':
        img = request.files["myfile"]
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_img.jpg'))
        return render_template('uploadImg.html',image='uploaded_img.jpg')
    else:
        gray_img = Image.open('project/static/uploaded_img.jpg').convert('L')
        gray_img_array = resize_image(gray_img=gray_img)
        gen_out = g_model.predict(gray_img_array)
        gen_out = (gen_out + 1) / 2.0
        gen_out = gen_out.reshape(256, 256, 3)
        plt.imshow(gen_out)
        plt.axis('off') 
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg'))
        plt.close()
        return render_template('uploadImg.html',image='uploaded_img.jpg',output='output.jpg')

if __name__ == '__main__':
    app.run()