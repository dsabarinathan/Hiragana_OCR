
import cv2
import math
import argparse
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import cv2
import json
import numpy as np
from tensorflow.keras import backend as K
from get_coordinate import get_object_coordinates

app = Flask(__name__)


class OCR():
    
    def __init__(self,path="/app/models/model-ocr-0.1829.h5",config_path="/app/config.json"):

        # Read the config JSON file
        with open(config_path, 'r',encoding="utf-8") as file:
            self.config_data = json.load(file)

        # Get the threshold value
        self.threshold = self.config_data['hiragana']['threshold']
        
        # Get the label dictionary
        self.label_dict = self.config_data['hiragana']['label']           
        
        # load the model from local        
        self.model = load_model(path,custom_objects={"K": K})

    def run(self,image):
        # extract the character coordinates using the cv2 contours
        coordinate,thresholdedImage = get_object_coordinates(image)
        
        image_batch = np.zeros((1,64,64,1))
        output =[]
        
        for row in range(len(coordinate)):
           temp = {}
           # crop the image 
           cropImg = thresholdedImage[coordinate[row][1]:coordinate[row][3],coordinate[row][0]:coordinate[row][2]]
           # resize the image 
           image_batch[0,:,:,0] = cv2.resize(cropImg,(64,64))*255
          
           # predict the results
           predict = self.model.predict(image_batch)
           position = np.argmax(predict)
           
           label_name = self.label_dict[str(position)]                    
           temp["text"] = label_name
           temp["coord"] = coordinate[row]  # Xmin,Ymin,Xmax,Ymax
           
           output.append(temp)
           
        return output
      
@app.route('/',methods=["POST"])
def index():

    image = request.files['file'].read()
    npimg = np.fromstring(image, np.uint8)
    image_np = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    
    results = ocrAPP.run(image_np)

    result_json={}
    result_json["result"] = results
    response =  jsonify(result_json)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response
if __name__ == '__main__':
    ocrAPP = OCR()
    app.run(debug=True, host='0.0.0.0')

           
