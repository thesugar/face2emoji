import numpy as np
from keras import models
from PIL import Image
from utils.glove_loader import GloveModel
from face_classifier import create_model
from pathlib import Path

def load_model(weight_path :str):
    if weight_path is None:
        raise ValueError('weight_path is undefined!')
    model = create_model()
    model.load_weights(weight_path)
    return model

def get_embedding_from_photo(image_clf=None, feeling_vector=None, photo_path=None):
    
    if photo_path is None:
        raise Exception('photo_path is undefined!')

    # load image
    input_image = Image.open(photo_path).resize((48, 48), Image.BICUBIC).convert('L')
    input_image = np.array(input_image)
    input_image = input_image.reshape(1, 1, 48, 48)
    result = image_clf.predict(input_image)
    print('---estimated feelings generated from photo are... ')
    print(result)
    
    result_vector = np.dot(result, feeling_vector)
    np.save('./vector/result_vector'+str(photo_path).replace('photo/','').replace('.jpeg',''), result_vector)
    
if __name__ == '__main__':
    print('start processing')
    model_path = # please designate the model path
    image_clf = load_model(model_path)
    
    # word embedding
    glove_model = GloveModel()
    print("now loading GloVe Model...")
    glove_model.load(data_dir_path='../emoji-gan/utils/glove.6B.300d.txt', embedding_dim=300)
    emb = glove_model.word2em
    print("Glove Model has been loaded!")
    
    feeling_vector = [emb['neutral'], emb['happiness'], emb['surprise'], emb['sadness'], emb['anger'], emb['disgust'], emb['fear'], emb['contempt']]
    
    photo_path = list(Path('photo/').glob('*.jpeg'))
    for photo in photo_path:
        get_embedding_from_photo(image_clf=image_clf, feeling_vector=feeling_vector, photo_path=photo)
    print('done!')


