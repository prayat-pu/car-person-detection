import numpy as np 
import cv2
import pickle
from scipy.ndimage.measurements import label


def prepare(frame,y,ws):
    #cut top and bottom of frame
    y_end = y + ws 
    strip = frame[y: y_end, :, :]
    
    return strip

def get_hog_feature(img):
    hog = cv2.HOGDescriptor()
    img_hist = hog.compute(cv2.resize(img,(64,128),cv2.INTER_AREA)).reshape(1,-1)

    filename_scaler = 'own_model/scaler_car_pedestrian_classifier.pkl'
    scaler = pickle.load(open(filename_scaler,'rb'))
    scaled_feature = scaler.transform(img_hist)
    return scaled_feature

def get_prediction(x):
    filename_model = 'own_model/rf_car_pedestrian_classifier.sav'
    rf_clf = pickle.load(open(filename_model, 'rb'))
    predict = rf_clf.predict(x)
    return int(predict[0])

def locate(frame,window_size,window_position):
    step_ = 31
    size_step = 64
    y, ws = window_position, window_size 
    strip = prepare(frame, y, ws)
    
    boxes = []
    
    x_end = (strip.shape[1] // size_step - 1) * size_step
    # print(strip.shape)

    label = ['background','people','car']

    for i in range(0,strip.shape[0],step_):
        for j in range(0,x_end,step_):
            y1 = i
            x1 = j
            w = size_step
            h = size_step
            roi = strip[y1:y1+h,x1:x1+w]
            if roi.shape[0] < size_step:
                break
            # while True:
            #     cv2.imshow('',roi)
            #     if cv2.waitKey(33) == 13:
            #         break
            scaled_x = get_hog_feature(roi)

            label_idx = get_prediction(scaled_x)
            # print('predict image: ',label[label_idx])
            if label_idx == 1 or label_idx == 2:
                boxes.append((x1,y1,w,h))
    return boxes,strip

# def box_boundaries(box):
#   x1, y1 = box[0], box[1]
#   x2, y2 = box[0] + box[2], box[1] + box[2]  
#   return x1, y1, x2, y2

# class HeatMap:

#   def __init__(self, frame, memory, thresh):
    
#     self.blank = np.zeros_like(frame[:, :, 0]).astype(np.int)
#     self.map = np.copy(self.blank)
#     self.thresholded_map = None
#     self.labeled_map = None
#     self.samples_found = 0
#     self.thresh = thresh
#     self.memory = memory
#     self.history = []

#   def reset(self):
#     self.map = np.copy(self.blank)
#     self.history = []

#   def do_threshold(self):
#     self.thresholded_map = np.copy(self.map)
#     self.thresholded_map[self.map < self.thresh] = 0
        
#   def get(self):
#     self.do_threshold()
#     self.label()
#     return self.map, self.thresholded_map, self.labeled_map
      
#   def remove(self, boxes):
#     for box in boxes: 
#       x1, y1, x2, y2 = box_boundaries(box)    
#       self.map[y1: y2, x1: x2] -= 1
      
#   def add(self, boxes): 
#     for box in boxes: 
#       x1, y1, x2, y2 = box_boundaries(box)
#       self.map[y1: y2, x1: x2] += 1

#   def update(self, boxes):
    
#     if len(self.history) == self.memory:
#       self.remove(self.history[0])
#       self.history = self.history[1:]
    
#     self.add(boxes)
#     self.history.append(boxes)

#   def label(self):
#     labeled = label(self.thresholded_map)
#     self.samples_found = labeled[1]
#     self.labeled_map = labeled[0]

#   def draw(self, frame, color = (0, 225, 0), thickness = 5):
    
#     this_frame = frame.copy()
#     _, _, this_map = self.get()
#     # print(self.samples_found)
    
#     for n in range(1, self.samples_found+ 1):
#       coords =  (this_map == n).nonzero()
#       xs, ys = np.array(coords[1]), np.array(coords[0])
#       p1 = (np.min(xs), np.min(ys))
#       p2 = (np.max(xs), np.max(ys))
#       cv2.rectangle(this_frame, p1, p2, color, thickness)
    
#     return this_frame

#   def show(self, frame, tdpi = 80):
      
#     mp, tmp, lmp = self.get()
#     labeled_img = self.draw(frame)
    
#     fig, ax = plt.subplots(1, 4, figsize = (15, 8), dpi = tdpi)
#     ax = ax.ravel()

#     ax[0].imshow(np.clip( mp, 0, 255), cmap = 'hot')
#     ax[1].imshow(np.clip(tmp, 0, 255), cmap = 'hot')
#     ax[2].imshow(np.clip(lmp, 0, 255), cmap = 'gray')
#     ax[3].imshow(labeled_img)

#     for i in range(4):
#       ax[i].axis('off')