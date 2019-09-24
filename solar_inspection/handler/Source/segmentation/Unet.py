import os
from .model import Unet_base_model, UResNet34
import json
import cv2
import numpy as np

import gc
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf

class Unet :
    """
        Load solar panel segmentation model : __init__()
        Returns segemented masks : predict()

    """
    def __init__(self, model_type, settings):

        self.Image_Height = settings.Image_Height
        self.Image_Width  = settings.Image_Width
        self.Channels     = settings.Channels

        if (model_type == "RGB") : #RGB
            self.model_path = settings.unet_path_rgb
            self.model = Unet_base_model(self.Image_Height,self.Image_Width, self.Channels, model_type);
        elif (model_type == "IR") : #Gray
            self.model_path = settings.unet_path_ir
            self.model = UResNet34(encoder_weights=True);
        else :
            assert False, ("Incorrect unet model type")

        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path);
        else :
            assert False, ("Check segmentation model path")

    def predict(self, img_list, resize = True, scale = True):
        """
                  Returns predictions over list of images
        """

        data_array = np.empty((len(img_list), self.Image_Height, self.Image_Width,self.Channels), dtype = 'float32')
        i = 0;
        for img in img_list:
            if resize:
                img = cv2.resize(img.copy(), dsize=(self.Image_Height, self.Image_Width),interpolation=cv2.INTER_LANCZOS4)
            d_img = img.copy()

            if scale:
                d_img = (img-np.min(img)) / (np.max(img) - np.min(img))

            print(d_img.shape)
            if d_img.ndim==2: # To be reviewed ? whats the need og this
                d_img = d_img[:,:,np.newaxis]

            data_array[i] = d_img
            i = i+1;

        output = self.model.predict(data_array)

        return output

    # Reset Keras Session
    def reset_keras(self):

        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()

        try:
            del classifier # this is from global space - change this as you need
        except:
            pass

        print(gc.collect()) # if it's done something you should see a number being outputted

            # use the same config as you used to create the session
            # config = tensorflow.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 1
            # config.gpu_options.visible_device_list = "0"
            # set_session(tensorflow.Session(config=config))

        #Hard clean memory : to be reviewed
        # from numba import cuda
        # cuda.select_device(0)
        # cuda.close()
