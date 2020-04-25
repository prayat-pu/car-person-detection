import cv2
import numpy as np 
import pickle

def get_hog_feature(img):
    hog = cv2.HOGDescriptor()
    img_hist = hog.compute(img).reshape(1,-1)
    return img_hist

def main():
    # y = '2' y of image that use to predict
    path_image_test = 'Data/pedestrian_data/positive (64X64)/person000018.png'
    img = cv2.imread(path_image_test)
    #resize 
    img = cv2.resize(img,(64,128))
    #get hog feature'
    unscaled_x = get_hog_feature(img)

    
    # load the model and scaler from disk
    filename_model = 'own_model/LinearSvm_car_pedestrian_classifier.sav'
    filename_scaler = 'own_model/scaler_car_pedestrian_classifier.pkl'
    rf_clf = pickle.load(open(filename_model, 'rb'))
    scaler = pickle.load(open(filename_scaler,'rb'))

    #transform data
    scaled_x = scaler.transform(unscaled_x)
    labels = ['not people and car','car','people']
    print("class: ",labels[int(rf_clf.predict(scaled_x))])
    


if __name__ == "__main__":
    main()