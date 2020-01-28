from cgan_emoji import DCGAN
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

def generate_emoji_from_photo(vector_path=None, dcgan=None):

    result_vector = np.load(vector_path)
    generated_emoji = dcgan.generate_image_from_vector(result_vector, flag=False)
    generated_emoji = generated_emoji * 127.5 + 127.5
    generated_emoji = Image.fromarray(generated_emoji.astype(np.uint8))
    generated_emoji = generated_emoji.convert('RGB')
    
    generated_emoji.save('./images/fromphoto/' + str(vector_path).replace('vector/result_vector', '').replace('.npy', '') + '.png')

if __name__ == '__main__':
    
    img_path = './emoji/edited/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'
    dcgan = DCGAN(img_path, txt_path, glove_path)
    dcgan.load_model()
    
    vectors = list(Path('vector/').glob('*.npy'))

    for vec_path in vectors:
        generate_emoji_from_photo(vector_path=vec_path, dcgan=dcgan)

    print('emojis have been generated!')