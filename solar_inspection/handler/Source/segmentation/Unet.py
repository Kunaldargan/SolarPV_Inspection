import os
from .model import Unet_base_model, Unet_8
import json
import cv2
import numpy as np

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
        elif (model_type == "GRAY") : #Gray
            self.model_path = settings.unet_path_gray
        elif (model_type == "COM") : #combined
            self.model_path = settings.unet_path_combined
        else :
            assert False, ("Incorrect unet model type")


        self.model = Unet_base_model(self.Image_Height,self.Image_Width, self.Channels, model_type);
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path);
        else :
            assert False, ("Check segmentation model path")

    def predict(self, img_list):
        """
                  Returns predictions over list of images
        """
        data_array = np.empty((len(img_list), self.Image_Height, self.Image_Width,self.Channels), dtype = 'float32')
        i = 0;
        for img in img_list:
            #img = exposure.equalize_adapthist(img, clip_limit=0.03)
            im = cv2.resize(img.copy(), dsize=(self.Image_Height, self.Image_Width),interpolation=cv2.INTER_LANCZOS4)
            d_img = im.copy()
            #d_img = (im-np.min(im)) / (np.max(im) - np.min(im))

            print(d_img.shape)
            if d_img.ndim==2: # To be reviewed ? whats the need og this
                d_img = d_img[:,:,np.newaxis]

            data_array[i] = d_img
            i = i+1;

        output = self.model.predict(data_array)

        return output
