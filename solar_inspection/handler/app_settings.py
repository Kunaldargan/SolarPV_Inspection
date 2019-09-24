import json
import os
from solar_inspection.settings import BASE_DIR

class Settings_Parser:

    def __init__(self) :
        settings_path = os.path.join(BASE_DIR,"App_Setting.json");

        with open(settings_path, 'r') as set_file:
            settings= json.load(set_file);

        self.Image_Height =settings["Segmentation"]["unet_segmentation"]["height"];
        self.Image_Width =settings["Segmentation"]["unet_segmentation"]["width"];
        self.Channels =settings["Segmentation"]["unet_segmentation"]["channels"];
        self.unet_path_rgb = settings["Segmentation"]["unet_segmentation"]["rgb_model_path"];
        self.unet_path_ir = settings["Segmentation"]["unet_segmentation"]["ir_model_path"];
        self.unet_path_combined = settings["Segmentation"]["unet_segmentation"]["Multi_model_path"];
        self.seg_mode = settings["Segmentation"]["Mode"];
        self.cuttoff_percentage = settings["image_tools"]["cutoff_percentage"];
        self.media_path = settings["image_tools"]["media_path"];
        self.fault_detection_config = settings["yolo_fault_detection"]["config"];
        self.fault_detection_weights = settings["yolo_fault_detection"]["weights"];
        self.fault_detection_names_file = settings["yolo_fault_detection"]["names_file"];
        self.fault_detection_data_file = settings["yolo_fault_detection"]["data_file"];
        self.fault_detection_thresh = settings["yolo_fault_detection"]["detection_conf"];
        self.darknet_path = settings["yolo_fault_detection"]["darknet_path"]
