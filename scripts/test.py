import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, multi_gpu_model

import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from NTU_Loader import *

num_classes = 60
batch_size = 4
stack_size = 64
path = sys.argv[1]

class i3d_modified:
    def __init__(self, weights = 'rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top = True, weights= weights)
        
    def i3d_flattened(self, num_classes = 60):
        i3d = Model(inputs = self.model.input, outputs = self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(x)
        new_model  = Model(inputs = i3d.input, outputs = predictions)        
        return new_model

i3d = i3d_modified(weights = 'rgb_imagenet_and_kinetics')
model = i3d.i3d_flattened(num_classes = num_classes)
model.load_weights('../model/ntu-cv_pre-trained_rgb_model.hdf5')
optim = SGD(lr = 0.01, momentum = 0.9)

model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

test_generator = DataLoader_video('../split/test_NTU.txt', path, batch_size = batch_size)

results = np.argmax(model.predict_generator(generator = test_generator, workers = cpu_count()-2), axis=-1)
label_map = [i.strip() for i in open('labels.txt').readlines()]
actions = []
for video in results:
    actions.append(label_map[video])

with open('../output/results.txt', 'w') as f:
    for item in actions:
        print >> f, item
