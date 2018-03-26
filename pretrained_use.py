import numpy as np
import os
import time

from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet, preprocess_input

# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.vgg19 import VGG19, preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input


model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

start_time = time.time()

dir = "100ToTest/01Taxol(sharp)/"
new_dir = "MobileNet100ToTest/01Taxol(sharp)/"

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for file in os.listdir(dir):
    img_path = os.path.join(dir, file)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    to_save = np.ndarray.flatten(np.array(feature[0, :, :, :]))
    np.save(new_dir + str(file).split('.')[0], to_save)

end_time = time.time()

print("Whole time:", end_time - start_time)
