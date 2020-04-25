import cv2
import numpy as np 
import pickle
from new_sliding import *




def box_boundaries(box):
  x1, y1 = box[0], box[1]
  x2, y2 = box[0] + box[2], box[1] + box[3]  
  return x1, y1, x2, y2

def put_boxes(frame, boxes, color = (255, 0, 0), thickness = 5):
    
  out_img = frame.copy()

  for box in boxes:
    x1, y1, x2, y2 = box_boundaries(box)    
    cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)
    
  return out_img


def main():
    video_name = 'video_test/y2mate.com - G30 DVR CAR CAMERA Video Record_EECR-Mi3-SE_360p.mp4'
    video = cv2.VideoCapture(video_name)

    window_sizes = 180,170,160
    strip_positions = 90,80,100
    boxed_images, strips = [], []

    

    while True:
        _,frame = video.read()

        for ws,wp in zip(window_sizes, strip_positions):
            boxes,strip = locate(frame,ws,wp)
            box_img = put_boxes(strip,boxes)
            break

            
        cv2.imshow('Own model Detections',box_img)
        if cv2.waitKey(33) == 13:
            break

       


if __name__ == "__main__":
    main()