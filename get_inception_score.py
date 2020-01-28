import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from PIL import Image
from pathlib import Path
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                visible_device_list="0", # specify GPU number
                allow_growth=True)
        )
set_session(tf.Session(config=config))


inception_model = InceptionV3()

def calc_inception_score(images=None, batch_size=None):
    
    if images is None:
        raise Exception('please pass the array of images.')
    if batch_size is None:
        raise Exception('please designate the batch size')
        
    r = None
    n_batch = (images.shape[0] + batch_size - 1) // batch_size
    for j in range(n_batch):
        x_batch = images[j * batch_size:(j + 1) * batch_size, :, :, :]
        r_batch = inception_model.predict(preprocess_input(x_batch))
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    p_y = np.mean(r, axis=0) # p(y)
    e = r * np.log(r / p_y) # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1) # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e) # inception score
    
def transform_images(path=None, file_type=None):
    
    if path is None:
        raise Exception('please designate the path for the directory which contains the images you want to get Inception Score.')
    
    if file_type is None:
        raise Exception('please designate the type of images (i.e. ".jpeg", ".png")')
        
    x_list = []
    for image_ in Path(path).glob('*'+file_type):
        img = Image.open(image_).resize((299, 299), Image.BICUBIC)
        img = img.resize(size=(299, 299), resample=Image.BICUBIC)
        x_list.append(image.img_to_array(img))
    return np.array(x_list).astype('float32') / 127.5 - 1
    
if __name__ == '__main__':
    
    image_dir_path = sys.argv[1] # Inception Score を取得したい画像たちの格納先（'./photo/'）
    file_type = sys.argv[2] # Inception Scoreを取得したい画像の拡張子（'.jpeg'）
    batch_size = int(sys.argv[3])
    
    score = calc_inception_score(transform_images(image_dir_path, file_type=file_type), batch_size=batch_size)
    
    print("done!")
    print("score is {}".format(score))