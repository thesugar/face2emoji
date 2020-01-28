import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from nltk.tokenize import sent_tokenize
import re

def make_images_np():
    '''
    make numpy file of pixels from face photographs.
    (in other words, convert face photos into NumPy array)
    '''
    face_photo_dir = 'facephoto20200119/'
    photo_path = list(Path(face_photo_dir).glob('*.jpeg'))
    #photo_path = [str(filename).replace(face_photo_dir+'IMG_','').replace('.jpeg','') for filename in list(Path(face_photo_dir).glob('*.jpeg'))]
    
    image_list = []
    
    for path in sorted(photo_path):
        image = Image.open(path).resize((224, 224), Image.BICUBIC)
        image = np.array(image)
        #image = (image.astype(np.float32) / 127.5) - 1.0
        image_list.append(image)
    image_np = np.array(image_list)
    
    np.save(face_photo_dir+'224_224/image_224_224', image_np)
    
if __name__ == '__main__':
    make_images_np()