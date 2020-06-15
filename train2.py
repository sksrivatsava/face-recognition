import numpy as np
from PIL import Image 
import os
import cv2
import pickle
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(base_dir,"dataset2")
rec=cv2.face.LBPHFaceRecognizer_create()
ylabels=[]
xtrain=[]
curr_id=0
label_ids={}
for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("PNG") or file.endswith("JPG"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ","-")
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=curr_id
                curr_id+=1
            id_=label_ids[label]
            #print(label_ids)    
            pil_img=Image.open(path).convert("L")
            img_arr=np.array(pil_img,"uint8")
            #print(img_arr)
            faces=face_cascade.detectMultiScale(img_arr,scaleFactor=1.3,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=img_arr[y:y+h ,x:x+w]
                xtrain.append(roi)
                ylabels.append(id_)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)  
rec.train(xtrain,np.array(ylabels))
rec.save("trainner.yml") 