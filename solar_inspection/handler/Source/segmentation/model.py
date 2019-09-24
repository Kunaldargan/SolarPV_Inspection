# Keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from .ResnetUnet import *

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#K.set_session(sess)

def Unet_base_model(Image_Height=400, Img_Width=400, Channels=1, model_type="RGB"):

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

def UResNet34(input_shape=(None, None, 3), classes=1, decoder_filters=16, decoder_block_type='upsampling',
                       encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):

    backbone = build_resnet(input_tensor=None,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=False,
                         block_type='basic')
    backbone.name = 'resnet34'

    if encoder_weights == True:
        load_model_weights(weights_collection, backbone , dataset= 'imagenet', classes = 1, include_top=False)

    skip_connections = list([129, 74, 37, 5]) # for resnet 34
    model = build_unet(backbone, classes, decoder_filters,
                       skip_connections, block_type=decoder_block_type,
                       activation=activation, **kwargs)
    model.name = 'u-resnet34'

    #freeze_model(backbone)

    return model
