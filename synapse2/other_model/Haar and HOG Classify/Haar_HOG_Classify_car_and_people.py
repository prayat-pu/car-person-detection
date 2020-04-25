import cv2
import numpy as np
import time

def readfile(path_video_name):
    return cv2.VideoCapture(path_video_name,0)

def train_model(path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    car_cascade_model = cv2.CascadeClassifier(path)
    return hog, car_cascade_model
 

def preprocessed_image(frames):
    # frame = cv2.resize(frames,(640,480))
    #strip window
    start_y = int((frames.shape[0])*0.25)
    h = 180
    frames = frames[start_y:start_y+h]

    blured = cv2.medianBlur(frames,3)
    # gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # _,threshold = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    return blured

def Detect_object(cap,people_model,car_model):
    while True:
        # reads frames from a video
        ret, frames = cap.read()

        # preprocess frames
        frame = preprocessed_image(frames)

        # Detects pedestrians of different sizes in the input image
        boxs1, weights = people_model.detectMultiScale(frame, winStride=(4,4))
        cars = car_model.detectMultiScale(frame,1.1,1)
        # To draw a rectangle in each pedestrians
        for (x,y,w,h) in boxs1:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Person', (x + 6, y - 6), font, 0.5, (0,255,0), 1)
            # Display frames in a window
        
        for (x,y,w,h) in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Cars', (x + 6, y - 6), font, 0.5, (0,255,0), 1)
        
        # cv2.imshow('detection', frames)
        cv2.imshow('Haar HOG Detection',frame)

        # Wait for Enter key to stop
        if cv2.waitKey(33) == 13:
            cv2.destroyAllWindows()
            break


def main():
    cv2.startWindowThread()
    # 1.read video file
    video_name = '../../video_test/y2mate.com - G30 DVR CAR CAMERA Video Record_EECR-Mi3-SE_360p.mp4'
    cap = readfile(video_name)

    # 2.training model
    people_hog_model, car_haar_model = train_model('cars.xml')

    # 3.test with recorded video
    Detect_object(cap,people_hog_model,car_haar_model)



if __name__ == "__main__":
    main()