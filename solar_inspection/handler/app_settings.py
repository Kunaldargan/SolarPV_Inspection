import json
import os
from solar_inspection.settings import BASE_DIR

class Settings_Parser:

    def __init__(self) :
        settings_path = os.path.join(BASE_DIR,"App_Setting.json");

        with open(settings_path, 'r') as set_file:
            settings= json.load(set_file);

        self.Image_Height =settings["unet_segmentation"]["height"]
        self.Image_Width =settings["unet_segmentation"]["width"]
        self.Channels =settings["unet_segmentation"]["channels"]
        self.unet_path_rgb = settings["unet_segmentation"]["rgb_model_path"]
        self.unet_path_gray = settings["unet_segmentation"]["gray_model_path"]
        self.unet_path_combined = settings["unet_segmentation"]["Multi_model_path"]
