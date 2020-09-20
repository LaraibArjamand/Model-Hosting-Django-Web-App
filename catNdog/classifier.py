from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
#method to load model

#method to preprocess our image to
#make it ccompatible to our model input
def preprocess(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image