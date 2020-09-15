import os 
import json 
ori_datadir = ''
tar_datadir = ''
json_path = '/home/xieyu/project/meta-dataset/meta-dataset/Records/ilsvrc_2012/dataset_spec.json'

with open(json_path,'r') as load_f: 
    load_dict = json.load(load_f)
    print(load_dict.keys())