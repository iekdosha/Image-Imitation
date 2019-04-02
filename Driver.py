import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from PIL import Image
import json

'''
This simple program showcase traning a neural network to estimate how gray should a pixel be given its position.
The program then prints a representation of the image by presenting the estimation of each pixel value into a new image.

By Itzhak Eretz Kdosha
'''
# Load configuration
hl_sizes = []
img_path = ""
NO_OF_RANDOM_POINTS = None
LEARNING_RATE = None
NO_OF_ITERATIONS_BEFORE_REFRESH = None
EPOCH_COUNT = None
def loadConfig():
   global img_path, hl_sizes, output_path, NO_OF_RANDOM_POINTS, LEARNING_RATE, NO_OF_ITERATIONS_BEFORE_REFRESH,EPOCH_COUNT
   with open('config.json') as f:
      config = json.load(f)
      img_path = str(config['img_path'])
      for i in range(len(config['hidden_layers'])):
         hl_sizes.append(config['hidden_layers'][i])
      NO_OF_RANDOM_POINTS = config['no_of_random_points']
      LEARNING_RATE = config['learning_rate']
      NO_OF_ITERATIONS_BEFORE_REFRESH = config['no_of_iteration_before_refresh']
      EPOCH_COUNT = config['epoch_count']


loadConfig()

#load image
img = Image.open(img_path).convert('L')
WIDTH, HEIGHT = img.size

# helper functions to convert location on the image into a vector and the other way around
def to_loc(norm):
   return [int((norm[0]*0.5+0.5)*WIDTH),int((norm[1]*0.5+0.5)*HEIGHT)]

def to_norm(loc):
   return [2.0*loc[0]/WIDTH-1,2.0*loc[1]/HEIGHT-1]

#map image image to an ordered matrix
img_data = list(map( lambda v: v/255 , list(img.getdata())))
img_data = [img_data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

#generate reandom points in range [-1,1]
random_spots = None
correct_value = None

# helper function to refresh the random points dataset
def refresh_data():
   global random_spots
   global correct_value
   random_spots = np.random.rand(NO_OF_RANDOM_POINTS, 2) * 2 - 1

   # map correct values for randomed points
   correct_value = []
   for p in random_spots:
      p_ = to_loc(p)
      x = int(p_[0])
      y = int(p_[1])
      correct_value.append([img_data[y][x], 1 - img_data[y][x]])

   correct_value = np.array(correct_value)


# specify the size of the input and output layers
INPUT_LAYER_SIZE = 2
OUTPUT_LAYER_SIZE = 2

# placeholders for the input (X) and correct output (Y)
X = tf.placeholder(tf.float32, [None, INPUT_LAYER_SIZE])
Y = tf.placeholder(tf.float32)

# constructing the network variables and structure
NO_OF_HIDDEN_LAYERS = len(hl_sizes)
hl = {}
W = {}
b = {}

prev_size = INPUT_LAYER_SIZE
prev_layer = X
cur_size = None
for i in range(NO_OF_HIDDEN_LAYERS):
   cur_size = hl_sizes[i]
   W[i] = tf.Variable(tf.random_uniform([prev_size, cur_size], -1, 1))
   b[i] = tf.Variable(tf.random_uniform([cur_size], -1, 1))
   hl[i] = tf.sigmoid(tf.add(tf.matmul(prev_layer, W[i]), b[i]))
   prev_layer = hl[i]
   prev_size = cur_size


# appending the last output layer
output_W = tf.Variable(tf.random_uniform([prev_size, OUTPUT_LAYER_SIZE], -1, 1))
output_b = tf.Variable(tf.random_uniform([OUTPUT_LAYER_SIZE], -1, 1))
output_layer = tf.nn.softmax(tf.add(tf.matmul(prev_layer,output_W), output_b))

# calculate loss
loss = -tf.reduce_sum(Y * tf.log(output_layer))

# minimize loss
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

#helper function that prints the current understanding of the system about what value each pixel should have
def print_image(name):
   img = Image.new('RGB', (WIDTH, HEIGHT))
   pix_pos = []

   for y in range(HEIGHT):
      for x in range(WIDTH):
         pix_pos.append([x,y])

   pix_norm =  list(map(to_norm ,pix_pos ))
   colors = sess.run(output_layer, feed_dict={X: pix_norm})

   for i in range(len(pix_pos)):
      gray = int(colors[i][0]*255)
      color = (gray,gray,gray)
      img.putpixel((pix_pos[i][0],pix_pos[i][1]), color)


   img.save('image'+str(name)+'.png')

# iterate through the proccess (train the network)
with tf.Session() as sess:
   tf.global_variables_initializer().run()
   for i in range(EPOCH_COUNT):
      if i%100 == 0:
         refresh_data()
      # indicate progress and print image every 2500 epochs
      if i % 1000 == 0:
         print('Loss after %d runs: %f' % (i, sess.run(loss, feed_dict={X: random_spots, Y: correct_value})))
         print_image(i)

      sess.run(train_step, feed_dict={X: random_spots, Y: correct_value})

   print_image('_final')