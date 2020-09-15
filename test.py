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


BASE_PATH = '/home/xieyu/project/meta-dataset/meta-dataset/Records'
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/data_config.gin'
# 2
gin.parse_config_file(GIN_FILE_PATH)
# 3
# Comment out to disable eager execution.
tf.compat.v1.enable_eager_execution()
# 4
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

SPLIT = learning_spec.Split.TRAIN

BATCH_SIZE = 16
ADD_DATASET_OFFSET = True
dataset_records_path = os.path.join(BASE_PATH, 'ilsvrc_2012')
dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
dataset_batch = pipeline.make_one_source_batch_pipeline(
    dataset_spec=dataset_spec, batch_size=BATCH_SIZE, split=SPLIT,
    image_size=224)

for idx, ((images, labels), source_id) in iterate_dataset(dataset_batch, 1):
    import pdb; pdb.set_trace()
    print(images.shape, labels.shape)