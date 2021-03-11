import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import shutil


LABELS_PATH="labels/"
ANNOTATED_IMAGES_PATH="annotatedImages"
IMAGES_PATH="images/"
IMAGES_DEST_PATH = "dataset/images/"
LABELS_DEST_PATH = "dataset/labels/"

mask_category = {'with_mask':0,'without_mask':1,'mask_weared_incorrect':2}

0 if os.path.isdir(ANNOTATED_IMAGES_PATH) else os.mkdir(ANNOTATED_IMAGES_PATH)
0 if os.path.isdir(LABELS_PATH) else os.mkdir(LABELS_PATH)


def convert_data(size,box):
    dw = np.float32(1./int(size[0]))
    dh = np.float32(1./int(size[1]))
    x1 = int(box[0])
    x2 = int(box[2])
    y1 = int(box[1])
    y2 = int(box[3])
    w = x2 - x1
    h = y2 - y1
    x = x1 + (w/2)
    y = y1 + (h/2)

    x = x*dw
    y = y*dh
    w = w*dw
    h = h*dh

    return [x,y,w,h]



def write_labels(size,box,fileName,name):
    content = convert_data(size,box)
    fileName.write(f"{mask_category[name]} {content[0]} {content[1]} {content[1]} {content[1]}\n")
    return



def make_bounding_boxes(image,name,start_point,end_point,linewidth=1,fontScale=1,font=cv2.FONT_HERSHEY_SIMPLEX,line=cv2.LINE_AA,color=(255,0,0)):
    org = (int(start_point[0]),int(start_point[1]+10))
    image = cv2.rectangle(image,start_point,end_point,color,linewidth)
    image = cv2.putText(image,str(mask_category[name]),org,font,fontScale,color,linewidth,line)
    return



def parse_xml():
    for file in tqdm(os.listdir('annotations')):
        filepath = os.path.join('annotations',file)
        xmlData = ET.parse(filepath)
        
        height = xmlData.find('size').find('height').text
        width = xmlData.find('size').find('width').text
        depth = xmlData.find('size').find('depth').text
        
        filename = xmlData.find('filename').text
        imageFilePath = os.path.join(IMAGES_PATH,filename)
        image = cv2.imread(imageFilePath)
        
        textFileName = open(os.path.join(LABELS_PATH,filename.split(".")[0]+".txt"),"a")
        
        for object in xmlData.findall('object'):
            name = object.find('name').text
            category=mask_category[name]
            xmin = object.find('bndbox').find('xmin').text
            ymin = object.find('bndbox').find('ymin').text
            xmax = object.find('bndbox').find('xmax').text
            ymax = object.find('bndbox').find('ymax').text
            
            start_point=(int(xmin),int(ymin))
            end_point = (int(xmax),int(ymax))
            make_bounding_boxes(image,name,start_point,end_point)
            
            size = [width,height]
            box = [xmin,ymin,xmax,ymax]
            write_labels(size,box,textFileName,name)
        
        textFileName.flush()
        textFileName.close()
        cv2.imwrite(os.path.join(ANNOTATED_IMAGES_PATH,filename),image)



def copy_data(file_list,category):
    images_dest_path = os.path.join(IMAGES_DEST_PATH,category)
    labels_dest_path = os.path.join(LABELS_DEST_PATH,category)

    if os.path.isdir(images_dest_path):
        print(f"{images_dest_path} already exists")
    else:
        os.makedirs(images_dest_path)

    if os.path.isdir(labels_dest_path):
        print(f"{labels_dest_path} already exists")
    else:
        os.makedirs(labels_dest_path)
    
    print("-----------creating {} set-----------------------".format(category))
    
    for file in tqdm(file_list):
        file_name = file.split(".")[0]
        shutil.copy2(IMAGES_PATH + file_name + ".png",os.path.join(images_dest_path,file_name+'.png'))
        shutil.copy2(LABELS_PATH + file_name + ".txt",os.path.join(labels_dest_path,file_name+'.txt'))



def split_data():
    image_list = os.listdir('images')
    
    train_list,test_list = train_test_split(image_list,test_size=0.2,random_state=7)
    val_list,test_list = train_test_split(test_list,test_size=0.5,random_state=7)
    
    copy_data(train_list,"train")
    copy_data(val_list,"test")
    copy_data(test_list,"val")



if __name__ =="__main__":
    print("---------------PARSING OF XML FILE AND CREATION OF LABELS AND ANNOTATED IMAGES STARTED-----------------")
    parse_xml()
    print("---------------PARSING OF XML FILE AND CREATION OF LABELS AND ANNOTATED IMAGES COMPLETED-----------------")
    print("\n")
    print("-----------------SPLITTING DATA INTO TRAIN,TEST AND VALIDATION SET--------------------")
    split_data()
    print("--------------SPLITTING OF DATA COMPLETED--------------------")
    
