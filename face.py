import cv2
import os




face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
os.chdir(r"C:\Users\srivatsava\Desktop\facedetection\dataset2")

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
name=input("enter name")
print(os.getcwd())

#print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
cam = cv2.VideoCapture(1)
while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(name+str(count)+"."+str(face_id)+".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
   

# Do a bit of cleanup

cam.release()
cv2.destroyAllWindows()