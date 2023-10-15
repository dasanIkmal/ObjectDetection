import uuid

import cv2
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)

path = "predict/sampleImages"
images = []
ImageList = os.listdir(path)
print("total Images", len(ImageList))

cap = cv2.VideoCapture('predict/111.mp4')

for img in ImageList:
    imgDir = cv2.imread(f'{path}/{img}',cv2.COLOR_BGR2GRAY)
    # endImage = cv2.cvtColor(imgDir,cv2.COLOR_BGR2GRAY)
    images.append(imgDir)

def findImageDes(images):
    des_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        des_list.append(des)
    return des_list

def findMatching(frame,des_list,treshold=120):
    kp2,des2=orb.detectAndCompute(frame,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        if des2 is not None:
            for des in des_list:
                des = des.astype(np.float32)
                des2 = des2.astype(np.float32)
                bf =cv2.BFMatcher()
                good = []
                matches = bf.knnMatch(des,des2,k=2)
                for m,n in matches:
                    if m.distance<0.75*n.distance:
                        good.append([m])
                print(len(good))
                if len(good)>treshold:
                    print("good score",len(good))
                    return True,len(good)
    except:
        pass
    return False,0


des_list = findImageDes(images)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    img2 =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    matchFound,score =findMatching(frame, des_list)
    if matchFound:
        cv2.imwrite('outputimages/'+ str(score) +'.jpg', frame)
        # break
    cv2.imshow("video",frame)
    key =cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()





