import numpy as np
import csv
import os
import random
import matplotlib.pyplot as plt

 

# List of folders for training, validation and test.
folder_names = {'train': 'FER2013Train',
                'val': 'FER2013Valid',
                'test': 'FER2013Test'}

labels_text = 'neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF'.split(',')

def get_next_batch(label_prefix_path, img_prefix_path,mode='train',batch_size=32, num_classes=7, max_sum_invalid_labels=1,
             min_major_feeling_score=5,verbose=False):
    # mode = train/val/test
    label_file_name = 'label.csv'
    f_name = folder_names[mode]
    label_path = os.path.join(label_prefix_path, f_name)
    label_path = os.path.join(label_path, label_file_name)
    img_path = os.path.join(img_prefix_path, f_name)
    if verbose: print('-----'+mode+'-----')
    data = get_data(label_path,num_classes,max_sum_invalid_labels,min_major_feeling_score,verbose)
    # shuffle the filtered data
    random.shuffle(data)

    start = 0
    while start<len(data):
        end = start+batch_size
        end = min(len(data)-1,end)
        # for specific case of only 1 example
        if len(data) == 1:
            end = 1

        batch = data[start:end]
        batch_x = [plt.imread(os.path.join(img_path, row[0])) for row in batch]
        batch_y = [row[1:] for row in batch]
        yield np.array(batch_x),np.array(batch_y)
        start = end+1



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
