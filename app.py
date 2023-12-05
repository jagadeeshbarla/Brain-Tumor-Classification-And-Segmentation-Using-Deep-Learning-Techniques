from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import io
import os
import cv2
import keras
from PIL import Image
import numpy as np
import base64
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import FileUpload
from IPython.display import display
app = Flask(__name__)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
segmentation_model=tf.keras.models.load_model('deeplabnetown.h5', custom_objects={'dice_loss':                   
dice_loss,'dice_coef':dice_coef})
model1=tf.keras.models.load_model('resnetfinal (3).h5',compile=False)
model2=tf.keras.models.load_model('inceptionfinal.h5',compile=False)
adam = keras.optimizers.Adam(learning_rate=0.0001)
model1.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics=['accuracy'])
H = 256
W = 256
save_image_path='mini code'
def read_image(image):
     ## [H, w, 3]
    image = cv2.resize(image, (W, H))       ## [H, w, 3]
    x = image/255.0                         ## [H, w, 3]
    x = np.expand_dims(x, axis=0)
    y_pred = segmentation_model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred * 255
    return y_pred
def majority_voting(model1, model2,image,image_size=224): 
    image = cv2.bilateralFilter(image, 2, 50, 50) 
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) 
    image = cv2.resize(image, (image_size, image_size))
    image = np.array(image) / 255.0
    image = np.reshape(image,[1,224,224,3])
    prediction1 = model1.predict(image)
    predictions2 = model2.predict(image)  
    majority_vote = np.round((prediction1 + predictions2) / 2)
    tumor=np.argmax(majority_vote, axis=1)
    if tumor==0:
        predictiontext="Tumor found and it is GLIOMA"
    elif tumor==1:
        predictiontext="tumor found and it is MENINGIOMA"
    elif tumor==2:
        predictiontext="No tumor found"
    else:
        predictiontext="Tumor found and it is PITUITARY"
    return predictiontext


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        image = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        y_pred=read_image(image)
        predictiontext=majority_voting(model1, model2,image)
        input_buffer = io.BytesIO()
        output_buffer = io.BytesIO()
        input_pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output_pil_image = Image.fromarray(y_pred)
        input_pil_image.save(input_buffer, format='JPEG')
        output_pil_image.save(output_buffer, format='JPEG')
        input_image = base64.b64encode(input_buffer.getvalue()).decode('utf-8')
        mask_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        # Render the index template with the input and output images
        return render_template('index.html', input_image=input_image, mask_image=mask_image,prediction_text=predictiontext)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)