"""
Tensorflow implementation of a neural network that paints an image.

The neural network learns the RGB output of each pixel using only
the (x, y) coordinates of that pixel.

Author: guiferviz
"""

import argparse
import os
import sys
import uuid
import re

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation

from scipy.misc import imresize


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h



# Parse args.
parser = argparse.ArgumentParser(description="""
Tensorflow implementation of a neural network that paints an image.

The neural network learns the RGB output of each pixel using only
the (x, y) coordinates of that pixel.

Author: guiferviz
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('input_file', nargs='?', type=str, default='input/rubik.jpg',
                    metavar='INPUT_FILE',
                    help='Path to the image to paint.')
parser.add_argument('-s', '--size', dest='size', nargs=2, type=int, default=(100, 100),
                    metavar=('WIDTH', 'HEIGHT'),
                    help='Size of the image.')
parser.add_argument('-n', '--net', dest='hidden_net', nargs='+', type=int, default=[64, 64, 64, 64, 64, 64],
                    metavar='HIDDEN_LAYERS',
                    help='List of number of neurons of the hidden layers.')
parser.add_argument('-c', '--no-compare', dest='no_compare', action='store_true',
                    help='Hide the original image on the left.')
args = parser.parse_args()


img_name = args.input_file
img = plt.imread(img_name)
img = imresize(img, (args.size[1], args.size[0]))

# Show initial image.
#plt.axis("off")
#plt.imshow(img, interpolation="nearest")
#plt.show()

# Generate dataset.
size = np.prod(img.shape[0:2])
train_X = np.zeros((size, 2))
train_Y = np.zeros((size, 3))

idx = 0
for i in np.ndindex(img.shape[0:2]):
    train_X[idx, :] = i
    train_Y[idx, :] = img[i[0], i[1], 0:3]
    idx += 1

# Normalizing the input.
train_X = (train_X - np.mean(train_X)) / np.std(train_X)

# Prepare output folder.
output_folder = "output/" + \
    img_name.split("/")[-1] + "_" + str(uuid.uuid4())
os.makedirs(output_folder)

# Create network.
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

n_neurons = [2,] + args.hidden_net + [3,]
with open("%s/readme.txt" % output_folder, "w") as text_file:
    str_net = ",".join(str(i) for i in n_neurons)
    print "Network architecture:", str_net
    text_file.write(str_net)

current_input = X
for layer_i in range(1, len(n_neurons)):
    current_input = linear(
        X=current_input,
        n_input=n_neurons[layer_i - 1],
        n_output=n_neurons[layer_i],
        activation=tf.nn.relu if layer_i + 1 < len(n_neurons) else None,
        scope='layer_' + str(layer_i))
pred_Y = current_input

#cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred_Y - Y), 1))
cost = tf.reduce_mean(tf.reduce_sum(tf.abs(pred_Y - Y), 1))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cost)

plt.figure()
if not args.no_compare:
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(img, interpolation="nearest")
    plt.subplot(122)
plt.axis("off")
img_plot = plt.imshow(img, interpolation="nearest")
plt.tight_layout()
plt.subplots_adjust(top=0.85)

batch_size = 50
total_it = 0
iterations_per_paint = 1  # updates the predicted image once each iteration
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    def update_fig(*args):
        global total_it

        for it_i in range(iterations_per_paint):
            idxs = np.random.permutation(size)
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: train_X[idxs_i], Y: train_Y[idxs_i]})

            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print(total_it, training_cost)
            total_it += 1

        outimg = pred_Y.eval(feed_dict={X: train_X}, session=sess)
        outimg = np.clip(outimg.reshape(img.shape), 0, 255).astype(np.uint8)
        img_plot.set_data(outimg)
        output_file_it = "%s/%07d.jpg" % (output_folder, total_it)
        # Saves the predicted image.
        #plt.imsave(output_file_it, outimg)
        # Saves the entire figure.
        plt.savefig(output_file_it, format="jpg")

        return img_plot,

    fig = plt.gcf()
    ani = animation.FuncAnimation(fig, update_fig, interval=1, blit=True)
    plt.show()
