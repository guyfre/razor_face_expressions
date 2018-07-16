import numpy as np
import csv
import os
import random
import matplotlib.pyplot as plt

 

# List of folders for training, validation and test.
folder_names = {'train': 'FER2013Train',
                'val': 'FER2013Valid',
                'test': 'FER2013Test'}

labels_text = 'neutral, happiness,surprise, sadness, anger, disgust, fear, contempt, unknown, NF'.split(',')

def get_next_batch(label_prefix_path, img_prefix_path,mode='train',batch_size=32, num_classes=7, max_sum_invalid_labels=1,
             min_major_feeling_score=5):
    # mode = train/val/test
    label_file_name = 'label.csv'
    f_name = folder_names[mode]
    label_path = os.path.join(label_prefix_path, f_name)
    label_path = os.path.join(label_path, label_file_name)
    img_path = os.path.join(img_prefix_path, f_name)

    data = get_data(label_path,num_classes,max_sum_invalid_labels,min_major_feeling_score)
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
        yield batch_x,batch_y
        start = end+1




# def get_data_dicts(folder_names, label_prefix_path, img_prefix_path, num_classes=7, max_sum_invalid_labels=1,
#              min_major_feeling_score=5):
#     label_file_name = 'label.csv'
#
#     f_name = folder_names['train']
#     label_path = os.path.join(label_prefix_path,f_name)
#     label_path = os.path.join(label_path,label_file_name)
#     img_path = os.path.join(img_prefix_path,f_name)
#
#     train_dict = get_data(label_path, img_path, num_classes, max_sum_invalid_labels, min_major_feeling_score)
#
#     f_name = folder_names['val']
#     label_path = os.path.join(label_prefix_path, f_name)
#     label_path = os.path.join(label_path, label_file_name)
#     img_path = os.path.join(img_prefix_path, f_name)
#
#     val_dict = get_data(label_path, img_path, num_classes, max_sum_invalid_labels, min_major_feeling_score)
#
#     f_name = folder_names['test']
#     label_path = os.path.join(label_prefix_path, f_name)
#     label_path = os.path.join(label_path, label_file_name)
#     img_path = os.path.join(img_prefix_path, f_name)
#
#     test_dict = get_data(label_path, img_path, num_classes, max_sum_invalid_labels, min_major_feeling_score)
#
#     return train_dict,val_dict,test_dict


def get_data(label_path, num_classes=7, max_sum_invalid_labels=1, min_major_feeling_score=5):
    data = []
    with open(label_path, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            # remove the size of image and convert elements to int except the file name
            row = row[0:1] + map(int, row[2:])
            data.append(row)
    size_before = len(data)
    print('before filtering there were ' + str(size_before) + ' examples')
    # filter and keep only examples which contempt+unknown+NF<2 and at least one of the other feeling>5
    data = [row for row in data if
            sum(row[-3:]) <= max_sum_invalid_labels and any(i >= min_major_feeling_score for i in row[1:])]
    size_after = len(data)
    print('after filtering there are ' + str(size_after) + ' examples')
    print('dropped %.2f%%') % (float(size_before - size_after) / size_before * 100)
    class_counters = np.zeros(num_classes)
    # statistics
    for row in data:
        ind = np.array(row[1:]).argmax()
        class_counters[ind] += 1
    print zip(labels_text, class_counters)

    data = [row[:num_classes + 1] for row in data]
    return data

    # # generate a dictionary with the image
    # data_dict = {}
    # for row in data:
    #     file_name = row[0]
    #     file_path = os.path.join(img_path, file_name)
    #     with open(file_path, 'rb') as img_file:
    #         image_b = img_file.read()
    #         data_dict[file_name] = {}
    #         data_dict[file_name]['image'] = image_b
    #         data_dict[file_name]['labels'] = np.array(row[1:num_classes + 1])
    # return data_dict



# img_path = '/home/rocket/PycharmProjects/test/FERPlus/data_base_dir'
# label_path = '/home/rocket/PycharmProjects/test/FERPlus/data'
# get_data(folder_names,label_path,img_path)