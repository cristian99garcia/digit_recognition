#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
from PIL import Image


class NetShape:
    INPUT = 784     # 28 * 28
    HIDDEN1 = 512
    HIDDEN2 = 256
    HIDDEN3 = 128
    OUTPUT = 10     # 10 posibles digitos


learning_rate = 0.0001
n_iterations = 1000
n_steps = n_iterations // 10
batch_size = 128
dropout = 0.01

X = tf.placeholder("float", [None, NetShape.INPUT])
Y = tf.placeholder("float", [None, NetShape.OUTPUT])
keep_prob = tf.placeholder(tf.float32)


class Weight:
    W1 = tf.Variable(tf.truncated_normal([NetShape.INPUT, NetShape.HIDDEN1], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([NetShape.HIDDEN1, NetShape.HIDDEN2], stddev=0.1))
    W3 = tf.Variable(tf.truncated_normal([NetShape.HIDDEN2, NetShape.HIDDEN3], stddev=0.1))
    OUT = tf.Variable(tf.truncated_normal([NetShape.HIDDEN3, NetShape.OUTPUT], stddev=0.1))


class Bias:
    B1 = tf.Variable(tf.constant(0.1, shape=[NetShape.HIDDEN1]))
    B2 = tf.Variable(tf.constant(0.1, shape=[NetShape.HIDDEN2]))
    B3 = tf.Variable(tf.constant(0.1, shape=[NetShape.HIDDEN3]))
    OUT = tf.Variable(tf.constant(0.1, shape=[NetShape.OUTPUT]))


layer_1 = tf.add(tf.matmul(X, Weight.W1), Bias.B1)
layer_2 = tf.add(tf.matmul(layer_1, Weight.W2), Bias.B2)
layer_3 = tf.add(tf.matmul(layer_2, Weight.W3), Bias.B3)
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, Weight.OUT) + Bias.OUT

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)


for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    session.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    if i % n_steps == 0:
        minibatch_loss, minibatch_accuracy = session.run(
            [cross_entropy, accuracy],
            feed_dict={
                X: batch_x,
                Y: batch_y,
                keep_prob: 1.0,
            }
        )
        print("Iteración", str(i), "\t| Pérdida =", str(minibatch_loss), "\t| Precisión =", str(minibatch_accuracy))

test_accuracy = session.run(accuracy, feed_dict={
    X: mnist.test.images,
    Y: mnist.test.labels,
    keep_prob: 1.0
})
print("\nPrecisión:", test_accuracy)


test_images = [
    "image0-01.png",
    "image1-01.png",
    "image2-01.png",
    "image3-01.png",
    "image4-01.png",
    "image5-01.png",
    "image6-01.png",
    "image7-01.png",
    "image8-01.png",
    "image9-01.png",
]

for name in test_images:
    img = np.invert(Image.open(name).convert('L')).ravel()
    prediction = session.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
    print("Predicción para %s:" % name, np.squeeze(prediction))
