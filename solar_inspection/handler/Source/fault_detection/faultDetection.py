from pydarknet import Detector, Image
import shutil
import cv2
import os

class FaultDetection:
    """
    Input: Images or List of Images

    Output: Json file with information about faults and their location

    Functions:
    classify(): classifies in multiple classes of faults, returns fault dictionary for image list
    """
    def __init__(self, settings):
        self.settings = settings;

    def fault_detection(self, data):
        self.faults = {}
        if type(data) == list:
        	self.img_list = data
        elif type(data) == numpy.ndarray:
        	self.img_list = list(data)
        else :
        	assert False, ('Wrong type of input')

        #self.cutoff = float(settings.fault_detection_thresh)
        #print("using fault detection confidence threshold : ", self.cutoff)
        self.net = Detector(bytes(self.settings.fault_detection_config, encoding="utf-8"), bytes(self.settings.fault_detection_weights, encoding="utf-8"), 0, bytes(self.settings.fault_detection_data_file,encoding="utf-8"))

        detections = []
        for im in self.img_list:
            img_darknet = Image(im)
            results = self.net.detect(img_darknet)
            det = []
            for cat, score, bounds in results:
                det.append((bounds,cat))
                #if score > self.cutoff:
                x, y, w, h = bounds
                cv2.rectangle(im, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), thickness=2)
                cv2.putText(im,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0));
            cv2.imshow("window", im);
            cv2.waitKey(10)
            detections.append(det)

        return self.img_list, detections

    def fault_detect_valid(self, path_list = [], output_path="media"):
        #Uses system command and darknet valid : Path to darknet required
        if not os.path.exists(self.settings.darknet_path):
            assert False, ('Requires Darknet Library Path')

        #create temp test.txt
        with open(os.path.join(self.settings.media_path, "test.txt"), 'w') as f:
            for path in path_list:
                f.write(path+"\n");

        if not os.path.exists(self.settings.fault_detection_data_file):
            assert False, ('Requires Darknet Data File Path')

        file = open(self.settings.fault_detection_data_file,'r')
        data_file = file.readlines()

        #create data file with test.txt
        with open(os.path.join(self.settings.media_path, "test.txt"), 'w') as f:
            for path in path_list:
                f.write(path+'\n');

        #create data file with test.txt
        with open(os.path.join(self.settings.media_path, "temp.data"), 'w') as f_data:
            for data in data_file:
                if("valid" in data):
                    f_data.write("valid" + " = " + os.path.join(self.settings.media_path, "test.txt") + '\n')
                else:
                    f_data.write(data)

        temp_data = os.path.join(self.settings.media_path, "temp.data")

        pwd = os.getcwd()
        if os.path.exists(os.path.join(pwd,"results")):
            shutil.rmtree(os.path.join(pwd,"results"))
        os.mkdir(os.path.join(pwd,"results"))

        #Setup the paths
        darknetValid = os.path.join(self.settings.darknet_path, "darknet") +" detector valid " + temp_data + " " + self.settings.fault_detection_config + " " + self.settings.fault_detection_weights

        #Call detector valid command
        os.system(darknetValid)
        faults_classwise = {}
        for files in os.listdir(os.path.join(pwd,"results")):
            classname = files.split("_")[-1].split(".txt")[0]
            f = open(os.path.join(pwd,"results", files),'r')
            results = f.readlines()

            #annotate
            for out in results:
                det = out.rstrip().split(" ")
                filename = det[0]
                confidence, x, y, w, h = float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
                if filename not in faults_classwise:
                    faults_classwise[filename] = {}
                    faults_classwise[filename][classname] = [(confidence, x, y, w, h)]
                elif classname not in faults_classwise[filename]:
                    faults_classwise[filename][classname] = []
                    faults_classwise[filename][classname].append((confidence, x, y, w, h))
                else:
                    faults_classwise[filename][classname].append((confidence, x, y, w, h))

        print(faults_classwise)
        for filename, cls_list in faults_classwise.items():
            img = cv2.imread(os.path.join(self.settings.media_path, filename+".jpg"),1)
            width, height = img.shape[0], img.shape[1]
            for classname, dets in cls_list.items():
                for confidence, x, y, w, h in dets:
                    if confidence > self.settings.fault_detection_thresh:
                        print(x, y, w, h)
                        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), thickness=2)
                        cv2.putText(img,classname,(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0));
            print("Annotations path : ", os.path.join(output_path,'annotation_{}.jpg').format(str(filename)))
            cv2.imwrite(os.path.join(output_path,'annotation__{}.jpg'.format(str(filename))),img)

        return faults_classwise;
