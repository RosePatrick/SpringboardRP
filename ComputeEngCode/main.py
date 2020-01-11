import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

#define class labels
class_dict = {'c0':'safe driving', 'c1':'texting - right', 'c2': 'talking on the phone - right', 'c3': 'texting - left',
'c4': 'talking on the phone - left', 'c5': 'operating the radio', 'c6': 'drinking',
'c7': 'reaching behind', 'c8': 'hair and makeup', 'c9': 'talking to passenger'}

# 'content' is base-64-encoded image data.
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

if __name__ == '__main__':
  file_path = sys.argv[1]
  project_id = 'stable-hybrid-249623'
  model_id = 'ICN4772510494057073039'

  with open(file_path, 'rb') as ff:
    content = ff.read()

  prediction = get_prediction(content, project_id, model_id)
  #convert prediction received from model into readable results
  pred_label = prediction.payload[0].display_name
  lbl = get_label(pred_label)
  score = prediction.payload[0].classification.score

  #print label and score
  print("Prediction:",pred_label,"-",lbl)
  print("Score:",score)
