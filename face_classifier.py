from keras import layers
from keras import models
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                visible_device_list="2", # specify GPU number
                allow_growth=True)
        )
set_session(tf.Session(config=config))

K.set_image_data_format(data_format='channels_first')

def resblock(x, filter_num=64, kernel_size=(3, 3)):
    shortcut = x
    shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Conv2D(filter_num, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filter_num, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def create_model(filter_num=64, block_num=4):
    inputs = layers.Input(shape=[1, 48, 48])
    x = inputs
    
    for i in range(block_num):
        if i > 0:
            x = layers.MaxPooling2D((2,2))(x)
            filter_num = filter_num * 2
        filter_num = [64, 128, 256, 64][i]
        x = layers.Conv2D(filter_num, (3,3), padding="same")(x)
        x = resblock(x, filter_num)
        x = resblock(x, filter_num)
        x = resblock(x, filter_num)
    x = layers.GlobalAveragePooling2D()(x)
    y = layers.Dense(8, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
if __name__ == '__main__':

    X = np.load('facedataset.npy')
    label = pd.read_csv('fer2013new.csv')
    Y = label[label['Usage'] == 'Training']
    Y = Y.drop(['Usage', 'Image name', 'unknown', 'NF'], axis=1)
    del label
    
    X = (X.astype(np.float32) / 127.5) - 1
    
    X = X.reshape(-1, 1, 48, 48)

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        width_shift_range=0.2,  # 左右にずらす
        height_shift_range=0.2,  # 上下にずらす
        horizontal_flip=True  # 左右反転
    )

    baseSaveDir = './log/face20200110'
    checkpoint = os.path.join(baseSaveDir, 'classifier_{epoch:02d}epoch-loss{val_loss:.2f}-acc{val_acc:.2f}.hdf5')
    cp = ModelCheckpoint(filepath = checkpoint, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlogger = CSVLogger('./log/face20200110/history_face_classifier.csv')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    image_clf = create_model()
    image_clf.summary()

    history = image_clf.fit_generator(datagen.flow(X_train, y_train, batch_size=100),
                        steps_per_epoch=X_train.shape[0] // 100, epochs=1000,
                        callbacks=[reduce_lr, early_stopping, cp, csvlogger],
                        validation_data=(X_valid, y_valid))