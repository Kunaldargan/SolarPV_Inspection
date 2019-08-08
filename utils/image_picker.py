import cv2
import os 


#save in test folder 
cv_img = []
cv_img_2 = []
img_name = []

#input path
pathname1= '/home/galactica/Galactica/Datasets/Eric_6th_August_Gmail/2/U_of_U/Dataset_1/'

#output_path
output_path = '/home/galactica/Galactica/Datasets/Eric_6th_August_Gmail/2/U_of_U/4_Gaps/'

count = 0

for item1 in sorted(os.listdir(pathname1)):  
            #edit loop number for picking
            if((count%3) ==0):
                if item1.endswith('.jpg'):
                
                    path1 = os.path.join(pathname1,item1)
                    path2 = os.path.join(output_path, item1)
                    shutil.copyfile(path1,path2)  
            count+=1