
# Import Python and TensorFlow libraries
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
from random import shuffle
from tqdm import tqdm

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

DATA_PATH = 'C:\\ML_Data\\Dogs_v_Cats\\'
TRAIN_DIR = DATA_PATH + 'train'
TEST_DIR  = DATA_PATH + 'test'
LOG_DIR  = DATA_PATH + 'log'

IMAGE_SIZE = 50  # pics will be converted to a 50x50 Numpy grayscale array
LEARNING_RATE = 1e-3  # 0.001

TRAIN_DATA_NPY = DATA_PATH + 'train_data-{}-{}-{}-{}x{}.npy'.format(LEARNING_RATE,
                                                              '10conv-basic',
                                                              'adam',
                                                              IMAGE_SIZE, 
                                                              IMAGE_SIZE)

TEST_DATA_NPY  = DATA_PATH + 'test_data-{}-{}-{}-{}x{}.npy'.format(LEARNING_RATE,
                                                             '10conv-basic',
                                                             'adam',
                                                             IMAGE_SIZE, 
                                                             IMAGE_SIZE)

# remember which model was used
#
MODEL_NAME = 'Dogs_v_Cats-{}-{}-{}-{}x{}.model'.format(LEARNING_RATE,
                                                    '10conv-basic',
                                                    'adam',
                                                    IMAGE_SIZE,
                                                    IMAGE_SIZE)

# Label will be either: a dog or a cat
# 
# One-hot array: [x, x] will be all cat-ness [1,0] 
# or all dog-ness [0,1]
#
def label_image(image):
    word_label = image.split('.')[-3]  # back 3 for: 'cat.1.jpg' or 'dog'
    
    if word_label == 'cat' : return [1,0]
    elif word_label == 'dog' : return [0,1]

# create our training data, just need to call this once
# our labeled training data consists of 25k images of cats
# and dogs (12k5 each)
#
def create_training_data():
    training_data = []
    
    for image in tqdm(os.listdir(TRAIN_DIR)):
        label = label_image(image) # either: [1,0] or [0,1]
        path = os.path.join(TRAIN_DIR, image)

# convert the color image to grayscale
#
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
# then resize it
#
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        training_data.append([np.array(image), np.array(label)])

# now shuffle our data for randomness
#
    shuffle(training_data)

# save out our training data numpy array
#
    np.save(TRAIN_DATA_NPY, training_data)
    return training_data

# process our testing data
# with 1 being the highest predicted 
# result that a test image is of a dog
#
def process_testing_data():
    testing_data = []
    
    for image in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, image)
        image_num = image.split('.')[0]
        
# then resize it
#
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        testing_data.append([np.array(image), image_num])
    
    shuffle(testing_data)
    
# save out our testing data numpy array
#
    np.save(TEST_DATA_NPY, testing_data)
    
    return testing_data


# MAIN
#
# the following function for creating training
# data took:
# 100%|| 25000/25000 [01:09<00:00, 360.95it/s]
#
#training_data = create_training_data()

# if we've already saved the training data:
#
training_data = np.load(TRAIN_DATA_NPY)

# reset our TF variables, graphs
#
tf.reset_default_graph()

# Define our convolutional network layers
#
convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax') # output layer was 10 for MNIST
convnet = regression(convnet,
                     optimizer='adam', 
                     learning_rate=LEARNING_RATE,
                     loss='categorical_crossentropy',
                     name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = LOG_DIR)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model loaded: ", MODEL_NAME)

# training data will be all but the lastt 500
#
train = training_data[:-500]
test  = training_data[-500:]

# X is out labels, Y is our feature sets
#
X = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
test_y = [i[1] for i in test]

# now fit for five epochs:
#
model.fit({'input': X}, 
          {'targets': Y}, 
          n_epoch=5, 
          validation_set=({'input': test_x}, 
                          {'targets': test_y}),
          snapshot_step=500, 
          show_metric=True, 
          run_id=MODEL_NAME)

# Output results #1 for a 2CNN layer with 3 epochs
#
# Training Step: 1148  | total loss: 11.23158 | time: 165.619s
# | Adam | epoch: 003 | loss: 11.23158 - acc: 0.5122 -- iter: 24448/24500
# Training Step: 1149  | total loss: 11.29569 | time: 167.251s
# | Adam | epoch: 003 | loss: 11.29569 - acc: 0.5094 | val_loss: 11.92739 - val_acc: 0.4820 -- iter: 24500/24500
# --

# Output results #2 for a 6CNN layer with 5 epochs: 80% accuracy
#
# Training Step: 1914  | total loss: 0.42775 | time: 209.585s
# | Adam | epoch: 005 | loss: 0.42775 - acc: 0.8107 -- iter: 24448/24500
# Training Step: 1915  | total loss: 0.43835 | time: 211.553s
# | Adam | epoch: 005 | loss: 0.43835 - acc: 0.8047 | val_loss: 0.51669 - val_acc: 0.7380 -- iter: 24500/24500
                        
# run tensorboard
# tensorboard --logdir=foo:C:\ML_Data\Dogs_v_Cats\log


# Output results: 6conv layers: after re-running the model.
# fit a second instance with 10 epochs: 86% accuracy
#
# Training Step: 3829  | total loss: 0.34507 | time: 200.713s
# | Adam | epoch: 010 | loss: 0.34507 - acc: 0.8509 -- iter: 24448/24500
# Training Step: 3830  | total loss: 0.33831 | time: 202.364s
# | Adam | epoch: 010 | loss: 0.33831 - acc: 0.8565 | val_loss: 0.44616 - val_acc: 0.8060 -- iter: 24500/24500
#  

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model loaded: ", MODEL_NAME)

model.save(MODEL_NAME)

# run another 5 epochs
#
model.fit({'input': X}, 
          {'targets': Y}, 
          n_epoch=5, 
          validation_set=({'input': test_x}, 
                          {'targets': test_y}),
          snapshot_step=500, 
          show_metric=True, 
          run_id=MODEL_NAME)


# 6conv layers: after re-running the model.fit a second instance
# with 10 epochs 86% accuracy:
#
# Training Step: 3829  | total loss: 0.34507 | time: 200.713s
# | Adam | epoch: 010 | loss: 0.34507 - acc: 0.8509 -- iter: 24448/24500
# Training Step: 3830  | total loss: 0.33831 | time: 202.364s
# | Adam | epoch: 010 | loss: 0.33831 - acc: 0.8565 | val_loss: 0.44616 - val_acc: 0.8060 -- iter: 24500/24500
#  

# now, plot the first 20 test images
#
testing_data = process_testing_data()

# if we've already saved the training data:
#
#testing_data = np.load(TEST_DATA_NPY)

fig = plt.figure()

# plot first 12 testing images
# where: cat = [1,0], dog = [0,1]
#
for num, data in enumerate(testing_data[:20]):
    image_data = data[0]
    image_num  = data[1]
    
    y = fig.add_subplot(5, 4, num+1) # 3,4:12 or 5,4:20
    orig = image_data
    data = image_data.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label = 'Dog'  # is a dog?
    else: str_label = 'Cat'
    
    y.imshow(orig, cmap = 'gray')
    plt.title(str_label)
    
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
   
plt.show()


print("Done")
