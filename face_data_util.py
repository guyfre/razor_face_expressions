import numpy as np
import csv
import os
import random
import matplotlib.pyplot as plt
import cv2
import Image
import glob


 

# List of folders for training, validation and test.
folder_names = {'train': 'FER2013Train',
                'val': 'FER2013Valid',
                'test': 'FER2013Test'}

labels_text = 'neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF'.split(',')

def get_label_img_paths(label_prefix_path, img_prefix_path,mode='train'):
    label_file_name = 'label.csv'
    f_name = folder_names[mode]
    label_path = os.path.join(label_prefix_path, f_name)
    label_path = os.path.join(label_path, label_file_name)
    img_path = os.path.join(img_prefix_path, f_name)
    return label_path,img_path

def get_next_batch(label_prefix_path, img_prefix_path,mode='train',batch_size=32, num_classes=7, max_sum_invalid_labels=1,
             min_major_feeling_score=5,verbose=False):
    # mode = train/val/test
    # get the label and img paths
    label_path,img_path = get_label_img_paths(label_prefix_path,img_prefix_path,mode)

    if verbose: print('-----'+mode+'-----')
    data = get_data(label_path,num_classes,max_sum_invalid_labels,min_major_feeling_score,verbose)
    # shuffle the filtered data
    random.shuffle(data)

    start = 0
    while start<len(data):
        end = start+batch_size
        end = min(len(data),end)
        # for specific case of only 1 example
        if len(data) == 1:
            end = 1

        batch = data[start:end]
        batch_x = [plt.imread(os.path.join(img_path, row[0])) for row in batch]
        batch_y = [row[1:] for row in batch]
        yield np.array(batch_x),np.array(batch_y)
        start = end+1

def get_specific_batch(label_prefix_path, img_prefix_path,img_names,mode='train', num_classes=7):
    data=[]
    label_path,img_path = get_label_img_paths(label_prefix_path,img_prefix_path,mode)
    if img_names is None:
        return None
    with open(label_path, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] in img_names:
                # remove the size of image and convert elements to int except the file name
                row = row[0:1] + map(int, row[2:num_classes + 2])
                data.append(row)
    batch_x = [plt.imread(os.path.join(img_path, row[0])) for row in data]
    batch_y = [row[1:] for row in data]
    return np.array(batch_x),np.array(batch_y)


def get_data(label_path, num_classes=7, max_sum_invalid_labels=1, min_major_feeling_score=5,verbose=False):
    data = []
    with open(label_path, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            # remove the size of image and convert elements to int except the file name
            row = row[0:1] + map(int, row[2:])
            data.append(row)
    size_before = len(data)
    if verbose:print('before filtering there were ' + str(size_before) + ' examples')
    # filter and keep only examples which contempt+unknown+NF<2 and at least one of the other feeling>5
    data = [row for row in data if
            sum(row[-3:]) <= max_sum_invalid_labels and any(i >= min_major_feeling_score for i in row[1:])]
    size_after = len(data)
    if verbose:print('after filtering there are ' + str(size_after) + ' examples')
    if verbose:print('dropped %.2f%%') % (float(size_before - size_after) / size_before * 100)
    class_counters = np.zeros(num_classes)
    # statistics
    for row in data:
        ind = np.array(row[1:]).argmax()
        class_counters[ind] += 1
    if verbose:print zip(labels_text, class_counters)

    data = [row[:num_classes + 1] for row in data]
    return data


#--------------------------------------------------------
face_cascade='./haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(face_cascade)

def get_face(img_path=None,img=None,img_size=48,force_use=False):
    if not force_use:
        if img is None:
            img = cv2.imread(img_path)
        face = cascade.detectMultiScale(img)
        if len(face)==0:
            raise Exception('no face in image')
        x, y, w, h = face[0]
        sub_face = img[y:y + h, x:x + w]
        gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(img_size,img_size))
    return gray

# image_path = '/home/rocket/Downloads/b.jpg'
# def save_faces(cascade):
#     img = cv2.imread(image_path+image_file_name)
#     for i, face in enumerate(cascade.detectMultiScale(img)):
#         x, y, w, h = face
#         sub_face = img[y:y + h, x:x + w]
#         gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(image_path+'temp'+str(i)+'.jpg', gray)

# if __name__ == '__main__':
    # face_cascade = image_path+"haarcascade_frontalface_default.xml"
    # cascade = cv2.CascadeClassifier(face_cascade)
    # # Iterate through files
    # save_faces(cascade)


    # face = get_face(image_path)
    # plt.imshow(face)
    # plt.show()