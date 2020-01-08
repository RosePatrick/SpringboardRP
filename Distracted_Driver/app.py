import base64
import numpy as np
import io
from flask import request
from flask import jsonify
from flask import Flask
import sys
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

from utils import get_prediction, get_label

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    project_id = 'stable-hybrid-249623'
    model_id = 'ICN4772510494057073039'
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    prediction = get_prediction(decoded, project_id,  model_id)
    pred_label = prediction.payload[0].display_name
    lbl = get_label(pred_label)
    score = prediction.payload[0].classification.score

    response = {
        'prediction' : {
        'label' : lbl,
        'score' : score
        }
    }
    return jsonify(response)
