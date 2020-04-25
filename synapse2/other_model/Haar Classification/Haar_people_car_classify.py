import cv2
import numpy as np
import time
def readfile(path_video_name):
    return cv2.VideoCapture(path_video_name,0)

def train_model(path_input_file):
    model = cv2.CascadeClassifier(path_input_file)
    return model

def preprocessed_image(frames):
    start_y = int((frames.shape[0])*0.25)
    h = 180
    frames = frames[start_y:start_y+h]

    blured = cv2.medianBlur(frames,3)
    # gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    return blured


def Detect_object(cap,car_model,human_model):
    while True:
        # reads frames from a video
        ret, frames = cap.read()
        # convert to gray scale of each frames
        frame = preprocessed_image(frames)

        # Detects pedestrians of different sizes in the input image
        human = human_model.detectMultiScale(frame,1.1,1)
        cars = car_model.detectMultiScale(frame,1.1,1)

        # To draw a rectangle in each car
        for (x,y,w,h) in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Cars', (x + 6, y - 6), font, 0.5, (0,255,0), 1)

        # To draw a rectangle in each pedestrians
        for (x,y,w,h) in human:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Person', (x + 6, y - 6), font, 0.5, (0,255,0), 1)
            
        # Display frames in a window
        cv2.imshow('Only Haar detection', frame)

        # Wait for Enter key to stop
        if cv2.waitKey(33) == 13:
            break

def main():
    cv2.startWindowThread()
    # 1.read video file
    video_name = '../../video_test/y2mate.com - G30 DVR CAR CAMERA Video Record_EECR-Mi3-SE_360p.mp4'
    cap = readfile(video_name)

    # 2.training model
        #Data for training in .xml format
    human_xml_name = 'haarcascade_fullbody.xml'
    car_xml_name = 'cars.xml'
    human_model = train_model(human_xml_name)
    car_model = train_model(car_xml_name)

    # 3.test with recorded video
    Detect_object(cap,car_model,human_model)




if __name__ == "__main__":
    main()