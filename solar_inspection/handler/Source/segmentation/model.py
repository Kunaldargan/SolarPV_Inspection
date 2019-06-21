# Keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def Unet_base_model(Image_Height=400, Img_Width=400, Channels=1, model_type="RGB"):

    if model_type == "COM":

        input_layer = Input(shape=(Image_Height, Img_Width, Channels)) #x_train.shape[1:]

        #Encoder Network
        #Conv1
        conv1 = Conv2D(filters=32, kernel_size = (3,3), activation='relu', padding='same')(input_layer)
        #400 x 400
        batchnorm1 = BatchNormalization()(conv1)
        #400 x 400
        l1    = MaxPool2D(strides=(2,2))(batchnorm1)
        #200 x 200

        #Conv2
        conv2 = Conv2D(filters=32, kernel_size = (3,3), activation='relu', padding='same')(l1)
        #200 x 200
        batchnorm2 = BatchNormalization()(conv2)
        #200 x 200
        l2    = MaxPool2D(strides=(2,2))(batchnorm2);
        #100 x 100

        #Conv3
        conv3 = Conv2D(filters=64, kernel_size = (3,3), activation='relu', padding='same')(l2)
        #100 x 100
        batchnorm3 = BatchNormalization()(conv3)
        #100 x 100
        l3 = MaxPool2D(strides=(2,2))(batchnorm3)
        #50 x 50

        #Conv4
        conv4 = Conv2D(filters=64, kernel_size = (3,3), activation='relu', padding='same')(l3)
        #50 x 50
        batchnorm4 = BatchNormalization()(conv4)
        #50 x 50

        #Conv5
        conv5 = Conv2D(filters=128, kernel_size = (1,1), activation='relu', padding='same')(batchnorm4)
        #50 x 50
        batchnorm5 = BatchNormalization()(conv5)
        #50 x 50
        d1 = Dropout(0.5)(batchnorm5)

        concat1  = concatenate([UpSampling2D(size=(2,2))(d1),batchnorm3], axis=-1)

        #Decoder
        Up1   = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')(concat1)
        concat2  = concatenate([UpSampling2D(size=(2,2))(Up1),batchnorm2], axis=-1)
        Up2   = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(concat2)
        concat3  = concatenate([UpSampling2D(size=(2,2))(Up2),batchnorm1], axis=-1)
        Up3   = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(concat3)
        Up4   = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same')(Up3)

        d2 = Dropout(0.5)(Up4)
        output_layer = Conv2D(filters=1, kernel_size=(1,1),activation='sigmoid')(d2)

        model = Model(input_layer, output_layer)

        return model;

    else:
        input_layer = Input(shape=(Image_Height, Img_Width, Channels)) #x_train.shape[1:]

        #Encoder Network
        conv1 = Conv2D(filters=8, kernel_size = (3,3), activation='relu', padding='same')(input_layer)
        l1    = MaxPool2D(strides=(2,2))(conv1)
        conv2 = Conv2D(filters=16, kernel_size = (3,3), activation='relu', padding='same')(l1)
        l2    = MaxPool2D(strides=(2,2))(conv2)
        conv3 = Conv2D(filters=32, kernel_size = (3,3), activation='relu', padding='same')(l2)
        l3    = MaxPool2D(strides=(2,2))(conv3)
        conv4 = Conv2D(filters=32, kernel_size = (1,1), activation='relu', padding='same')(l3)

        concat1  = concatenate([UpSampling2D(size=(2,2))(conv4),conv3], axis=-1)

        #Decoder
        Up1   = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(concat1)
        concat2  = concatenate([UpSampling2D(size=(2,2))(Up1),conv2], axis=-1)
        Up2   = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(concat2)
        concat3  = concatenate([UpSampling2D(size=(2,2))(Up2),conv1], axis=-1)
        Up3   = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(concat3)
        Up4   = Conv2D(filters=64, kernel_size=(1,1), activation='relu', padding='same')(Up3)

        l4 = Dropout(0.5)(Up4)
        output_layer = Conv2D(filters=1, kernel_size=(1,1),activation='sigmoid')(l4)

        model = Model(input_layer, output_layer)

        return model

# Design our model architecture here
def Unet_8(img_width=256, img_height=256):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    '''
    n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    encodeds = []

    # encoder
    enc = inp
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Dropout(0.1*l_idx,)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)

    # decoder
    dec = enc
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Dropout(0.1*l_idx)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    return model
