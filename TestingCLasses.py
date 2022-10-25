import cv2
import FaceDetectionModule as fd #face detection module
import Classification as Classifier
import numpy as np
import math
import time
'''
There is problem occurs when we are going to collect the data. 
When we send the data to the classifier we have to crop the images into same sizes as it is easy
for classifier to classify the images with same size.

Solution to this is to add square in the back of cropped image.
'''

'''
Step 1. create directory with name Data and sub directories with classes like A
B, C and I LOVE YOU
Libraries required is opencv and mediapipe

Step 2: Crop the image when we get the hand.
Step 3: add to the white image for same size
Step 4: Collect the multiple images and assign specific class
Step 5: We are using google trainer named as teachable machine to train our data
        (https://teachablemachine.withgoogle.com/train)

'''

cam = cv2.VideoCapture(0)
'''
Sometimes it will not give you suggestions. Problem with newer version of opencv
So you need to install opencv version 4.5.4.60
'''
# Making an object of class handTracker
detectFace = fd.faceDetect()

# declare the classifier with model and lable
classifier = Classifier.Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imageSize = 300

folderName = "Data/C"
counter = 0

labels = ["afnan", "iqra", "kashan", "kashif", "nouman", "saba",
          "sadam", "saman", "shafique", "shazad", "sheryar", "talha",
          "touseef", "waleed","zia"]

while True:
    Success, frame = cam.read()
    finalFrame = frame.copy()

    face, bbox = detectFace.findFace(frame)


    # now we crop the hand image
    try:
        if Success:

            x, y, w, h = bbox

            # creating our own image for same size
            imgWhite = np.ones((300, 300, 3), np.uint8) * 255


            # staring hight ending hight, starting width and ending width
            # imgCrop = frame[y:y + h, x:x + w]
            imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape
            # add cropped image in to white image
            # imgWhite[0:imgCropShape[0], 0:imgCropShape[1]] = imgCrop

            # so in order to fit our croped image on the white image we have to do
            # some calculations

            aspectRatio = h / w  # if value is above one its mean hight is greater

            if aspectRatio > 1:         # fix the hight
                k = imageSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imageSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imageSize - wCal) / 2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(labels[index])

            else:                       # fix the width
                k = imageSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imageSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imageSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(labels[index])


            # cv2.imshow("Cropped Image", imgCrop)
            # cv2.imshow("WhiteImage", imgWhite)

            cv2.rectangle(finalFrame, (x - 20, y - 20), (x+w+20, y+h+20),
                          (255, 0, 255), 1)
            cv2.line(finalFrame, (x - 20, y - 20), (x - 20, y - 20 + 20), (255, 0, 255), 3)
            cv2.line(finalFrame, (x - 20, y - 20), (x - 20 + 20, y - 20), (255, 0, 255), 3)

            # cv2.line(img, (x + w, y), (x + w, y + 20), (0, 0, 0), 3)
            # cv2.line(img, (x + w, y), (w - 20, y), (0, 0, 0), 3)

            # cv2.line(finalFrame, (x  + w +20, y +20), (x + w+20, y +20 ), (255, 0, 255), 3)
            # cv2.line(finalFrame, (x  + w +20, y+20), (x+w - 20+20 , y +20), (255, 0, 255), 3)
            # cv2.rectangle(finalFrame, (x - offset, y - offset - 50),
            #              (x - offset + 90, y - offset - 50 + 50), (0, 255, 255), cv2.FILLED)
            cv2.putText(finalFrame, labels[index], (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)



        cv2.imshow("Webcam", finalFrame)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
    except:
        pass

cam.release()
cv2.destroyAllWindows()




