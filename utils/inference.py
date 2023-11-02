import tensorflow as tf
import numpy as np

def get_model(path='model/model.h5'):
    return tf.keras.models.load_model(path)

def infer(img):
    model=get_model()
    preds=model.predict(img[np.newaxis, :, :, :])
    return preds[0]