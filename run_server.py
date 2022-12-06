# import the necessary packages
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import img_to_array
import numpy as np
import flask
import io

import os
import json
from threading import Lock

# printing only warnings and error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

try:
    import tensorflow as tf
    from PIL import Image
    #ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    raise ImportError("ERROR: Failed to import libraries. Please refer to READEME.md file\n")

EXPORT_MODEL_VERSION = 1

from tf_example import TFModel

# initialize Flask application
app = flask.Flask(__name__)

@app.route("/", methods=["POST","GET"])
def index():

    if flask.request.method == "GET":
        return flask.render_template('index.html')
    data = {}

    # load image
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
       
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess the image and prepare it for classification
            # image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            # preds = model.predict(image)

            outputs = model.predict(image)
            print(f"Predicted: {outputs}")

            pred_label = outputs['predictions'][0]['label']
            pred_confidence = outputs['predictions'][0]['confidence']
            # if preds[0,0] > 0.5:
            #     result ="Normal Image"
            # else:
            #     result ="Abormal Image"                   
                       
            data["predictions result"] = pred_label#result
            data["predictions confidence"] = pred_confidence      
          
    # return the data 
    return flask.jsonify(data)

def prepare_image(image, target):

    # if the image mode is not three channels, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image / 255
 
    return image

# start the server
if __name__ == "__main__":
    print(("* Flask starting server..."
        "please wait until server has fully started"))
    global model

    dir_path = os.getcwd()
    
    model = TFModel(dir_path=dir_path)

    # model = load_model('Pepper_Bell_96.15_cnn_model.h5')
    app.run()
    
    
    