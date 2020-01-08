
import base64
import numpy as np
import io
import sys
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

#define class labels
class_dict = {'c0':'safe driving', 'c1':'texting - right', 'c2': 'talking on the phone - right', 'c3': 'texting - left',
'c4': 'talking on the phone - left', 'c5': 'operating the radio', 'c6': 'drinking',
'c7': 'reaching behind', 'c8': 'hair and makeup', 'c9': 'talking to passenger'}

#get image prediction from API
def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()
  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

#convert predicted label (e.g. c0, c1) to text (e.g. 'safe driving')
def get_label(pred):
    label = class_dict[pred]
    return label
