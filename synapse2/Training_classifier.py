from sklearn.svm import LinearSVC
import glob
import cv2
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def read_img_to_list(path,size):
    image_list = []
    print('load image... from ',path)
    for name in glob.glob(path):
        img = cv2.imread(name)
        resized_img = cv2.resize(img,(size[0],size[1]))
        image_list.append(resized_img)
    print('complete load image... from ',path)
    return image_list

def get_hog_feature(img_list):
    hog_list = []
    hog = cv2.HOGDescriptor()
    for img in img_list:
        hog_list.append(hog.compute(img).reshape(-1))
    return hog_list


def merge_dataset(dataset1,dataset2,dataset3):
    unsacled_x = np.vstack((dataset1,dataset2,dataset3)).astype(np.float64)
    scaler = StandardScaler().fit(unsacled_x)
    x = scaler.transform(unsacled_x)
    len_dataset1 = len(dataset1)
    len_dataset2 = len(dataset2)
    len_dataset3 = len(dataset3)
    y = np.hstack((np.ones(len_dataset1),np.zeros(len_dataset2),np.ones(len_dataset3)+1))    

    #save scaler
    filename = 'own_model/scaler_car_pedestrian_classifier.pkl'
    pickle.dump(scaler, open(filename, 'wb'))
    return x,y

def main():
    #read data and change image to vector use hog size 64,128
    path_vehicle = 'Data/vehicles/KITTI_extracted/*'
    path_nonvehicle = 'Data/non-vehicles/Extras/*'
    path_pedestrian = 'Data/pedestrian_data/positive (64X64)/*'

    vehicle_imgs = read_img_to_list(path_vehicle,(64,128))
    vehicle_hog_list = get_hog_feature(vehicle_imgs)

    nonvehicle_imgs = read_img_to_list(path_nonvehicle,(64,128))
    nonvehicle_hog_list = get_hog_feature(nonvehicle_imgs)

    pedestrian_imgs = read_img_to_list(path_pedestrian,(64,128))
    pedestrian_hog_list = get_hog_feature(pedestrian_imgs)
    
    # adjust dataset and train test split use test_size 20 percent
    x,y = merge_dataset(vehicle_hog_list,nonvehicle_hog_list,pedestrian_hog_list)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    

    # training data use Lienar svm model
    print('Training Model....')
    svm_clf = LinearSVC()
    svm_clf.fit(x_train, y_train)
    accuracy = svm_clf.score(x_test,y_test)
    print('accuracy: ',accuracy)
    # save the model to disk
    filename = 'own_model/LinearSvm_car_pedestrian_classifier.sav'
    pickle.dump(svm_clf, open(filename, 'wb'))



if __name__ == "__main__":
    main()