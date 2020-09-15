from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter
import gin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec
from meta_dataset.data import pipeline
import torch 
from PIL import Image 
import torchvision.transforms as transforms
import moco.loader

BASE_PATH = '/home/xieyu/project/meta-dataset/meta-dataset/Records'
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
# 2
gin.parse_config_file(GIN_FILE_PATH)
# 3
# Comment out to disable eager execution.
tf.compat.v1.enable_eager_execution()
# 4
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
augmentation = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
trans = transforms.Compose(augmentation)

to_torch_labels = lambda a: torch.from_numpy(a.numpy()).long()
to_torch_imgs = lambda a: torch.from_numpy(np.transpose(a.numpy(), (0, 3, 1, 2)))
def iterate_dataset(dataset, n):
  if not tf.executing_eagerly():
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
      for idx in range(n):
        yield idx, sess.run(next_element)
  else:
    for idx, episode in enumerate(dataset):
      if idx == n:
        break
      yield idx, episode
transform = moco.loader.TwoCropsTransform(trans)
def iterate_dataset_batch(dataset, num_batches, batch_size):
    if not tf.executing_eagerly():
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            for idx in range(num_batches):
                episode, source_id = sess.run(next_element)
                
                yield(to_torch_imgs(episode[0]), to_torch_labels(episode[1]))
    
    else: 
        batch_count = 0
        k_batch = []
        q_batch = []
        label_batch = []
        for idx, (episode, source_id) in enumerate(dataset):
            if batch_count == num_batches:
                break 
            images = to_torch_imgs(episode[0]).squeeze(0)
            
            images = transform(images)
            #batch_entry = [images[0], images[1], to_torch_labels(episode[1])]
            #curr_batch.append(batch_entry)
            k_batch.append(images[0])
            q_batch.append(images[1])
            label_batch.append(images[2])
            if len(curr_batch) == batch_size:
                images_q = torch.stack(q_batch)
                images_k = torch.stack(k_batch)
                labels = torch.stack(label_batch)
                k_batch = []
                q_batch = []
                label_batch = []
                batch_count += 1
                yield images_q, images_k, labels





SPLIT = learning_spec.Split.TRAIN

BATCH_SIZE = 1
ADD_DATASET_OFFSET = True
dataset_records_path = os.path.join(BASE_PATH, 'ilsvrc_2012')
dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
dataset_batch = pipeline.make_one_source_batch_pipeline(
    dataset_spec=dataset_spec, batch_size=BATCH_SIZE, split=SPLIT,
    image_size=224)

for images_q, images_k, labels in iterate_dataset_batch(dataset_batch, 10000, 256):
    # import pdb; pdb.set_trace()
    # images = images.numpy()
    # images = trans(images)
    # print(images.shape, labels.shape)
    print(images_q.shape)
    print(images_k.shape)