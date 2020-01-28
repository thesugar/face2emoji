# Xception model -> 300 classes
from keras import layers
from keras import models
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import LeakyReLU
from keras.optimizers import SGD
import os
import numpy as np
import pandas as pd

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                visible_device_list="3", # specify GPU number
                allow_growth=True)
        )
set_session(tf.Session(config=config))

K.set_image_data_format(data_format='channels_first')

'''
def resblock(x, filter_num=64, kernel_size=(3, 3)):
    shortcut = x
    #shortcut = layers.BatchNormalization()(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filter_num, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filter_num, kernel_size, padding="same")(x)
    x = layers.Add()([x, shortcut])
    return x
'''

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

'''
def create_model(filter_num=64, block_num=2):
    inputs = layers.Input(shape=[3, 299, 299])
    x = inputs
    
    for i in range(block_num):
        if i > 0:
            x = layers.MaxPooling2D((2,2))(x)
            filter_num = filter_num * 2
        filter_num = [128, 256][i]
        x = layers.Conv2D(filter_num, (3,3), padding="same")(x)
        x = resblock(x, filter_num)
        #x = resblock(x, filter_num)
        #x = resblock(x, filter_num)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(300, activation='softmax')(x)
    y = layers.Dense(300, activation='linear')(x)
    #x = LeakyReLU(alpha=0.01)(x)
    #y = layers.Dense(300)(x)

    model = models.Model(inputs=inputs, outputs=y)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.compile(loss=euclidean_distance_loss, optimizer=SGD(decay=1e-6, momentum=0.9, nesterov=True))
    return model
'''

def create_model():
    inputs = layers.Input(shape=[3, 224, 224])
    x = inputs
    x = layers.Lambda(lambda x: x / 255.)(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(300, activation='softmax')(x)
    y = layers.Dense(300, activation='linear')(x)
    model = models.Model(inputs=inputs, outputs=y)
    model.compile(loss=euclidean_distance_loss, optimizer=Adam())
    return model

if __name__ == '__main__':

    #X = np.load('facephoto20200119/image.npy', allow_pickle=True)
    X = np.load('facephoto20200119/224_224/image_224_224.npy')
    Y = np.load('./encoded_caption_of_emojis_200120.npy')
    #Y = np.load('cap_emb_pcato26dim.npy')
    
    #X = (X.astype(np.float32) / 127.5) - 1
    
    X = X.reshape(-1, 3, 224, 224)

    #X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)

    datagen = ImageDataGenerator(
        width_shift_range=0.2,  # 左右にずらす
        height_shift_range=0.2,  # 上下にずらす
        horizontal_flip=True  # 左右反転
    )

    baseSaveDir = './facephoto20200119/log/'
    checkpoint = os.path.join(baseSaveDir, '200122_{epoch:02d}epoch-loss{loss:.2f}.hdf5')
    cp = ModelCheckpoint(filepath = checkpoint, monitor='loss', save_best_only=True, mode='auto')
    csvlogger = CSVLogger('./facephoto20200119/log/history_face_classifier_200122.csv')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1)

    image_clf = create_model()
    image_clf.summary()
    '''
    history = image_clf.fit_generator(datagen.flow(X, Y, batch_size=60),
                        steps_per_epoch=X.shape[0] // 60, epochs=1000,
                        callbacks=[reduce_lr, early_stopping, cp, csvlogger],
                        #validation_data=(X_valid, y_valid))
    '''

    history = image_clf.fit(X, Y, batch_size=70, epochs=1000, callbacks=[reduce_lr, early_stopping, cp, csvlogger])