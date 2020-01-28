from cgan_emoji import DCGAN
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session, get_session

def generate_emoji_from_photo(result_vector=None, dcgan=None, index=None):

    generated_emoji = dcgan.generate_image_from_vector(result_vector, flag=False)
    generated_emoji = generated_emoji * 127.5 + 127.5
    generated_emoji = Image.fromarray(generated_emoji.astype(np.uint8))
    generated_emoji = generated_emoji.convert('RGB')
    
    result_save_dir = '/' # please designate the dir path on which you want to save the result emojis.
    
    generated_emoji.save(result_save_dir + str(index) + '.png')
    

if __name__ == '__main__':
    
    GPU_NUM = "3"
    
    config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(
                    visible_device_list=GPU_NUM, # specify GPU number
                    allow_growth=True)
            )
    set_session(tf.Session(config=config))
    
    img_path = './emoji/edited/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'
    dcgan = DCGAN(img_path, txt_path, glove_path)
    dcgan.load_model()
    
    # pca逆変換のために以下でpcaインスタンスを作る
    #cap_emb = np.load('encoded_caption_of_emojis_200120.npy')
    #from sklearn.decomposition import PCA
    # n_componentsに目標とする累積寄与率を指定
    #pca = PCA(n_components = 0.90)
    # fit_transformで主成分分析を行い、さらにデータを写像
    #cap_emb_pca = pca.fit_transform(cap_emb)
    # components_に主成分ベクトルが保存される
    
    
    # Generator に入れるベクトル（300次元）を指定する
    result_vector = np.load('hoge.npy') # please designate the vector path!
    
    for i, vec in enumerate(result_vector):
        vec = vec.reshape((1, 300))
        generate_emoji_from_photo(result_vector=vec, dcgan=dcgan, index=i)

    print('emojis have been generated!')