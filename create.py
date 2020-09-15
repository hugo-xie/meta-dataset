import os 
import json 
ori_datadir = '/home/xieyu/dataset/ImageNet/ILSVRC2012/train'
tar_datadir = '/home/xieyu/dataset/meta-dataset/imagenet/train'
json_path = '/home/xieyu/project/meta-dataset/meta-dataset/Records/ilsvrc_2012/dataset_spec.json'

with open(json_path,'r') as load_f: 
    load_dict = json.load(load_f)
    #print(len(set(list(load_dict['images_per_class']['TRAIN'].keys()))))
    count = 0
    imagenet_train = list(load_dict['images_per_class']['TRAIN'].keys())
    for subdir in imagenet_train:
        tem_dir_ori = os.path.join(ori_datadir, subdir)
        if os.path.exists(tem_dir_ori):
            #tem_dir_target = os.path.join(tar_datadir, subdir)
            #print(tem_dir_target)
            #os.mkdir(tem_dir_target)
            cmd = 'ln -s {} {}'.format(tem_dir_ori, tar_datadir)
            os.system(cmd)
            count += 1

    
    print(count)