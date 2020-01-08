# Distracted Driver Detection

The aim of this project is given the dataset of driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc), predict the likelihood of what the driver is doing in each picture. The 10 classes to predict are:
* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind
* c8: hair and makeup
* c9: talking to passenger

The model was implemented with Google Cloud Platform AutoML Vision and served with Flask micro web framework.
