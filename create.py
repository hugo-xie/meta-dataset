import os 
import json 
ori_datadir = '/home/xieyu/dataset/ImageNet/ILSVRC2012/train'
tar_datadir = ''
json_path = '/home/xieyu/project/meta-dataset/meta-dataset/Records/ilsvrc_2012/dataset_spec.json'

with open(json_path,'r') as load_f: 
    load_dict = json.load(load_f)
    #print(len(set(list(load_dict['images_per_class']['TRAIN'].keys()))))
    count = 0
    imagenet_train = list(load_dict['images_per_class']['TRAIN'].keys())
    for subdir in imagenet_train:
        if os.path.exists(os.path.join(ori_datadir, subdir)):
            count += 1
    
    print(count)