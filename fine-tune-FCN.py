import os
import scipy as scp
import scipy.misc

import matplotlib.pyplot as plt
import numpy as np
import logging
import tensorflow as tf
import sys
import fcn8_vgg
import utils
from datetime import datetime
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
from tensorflow.python.framework import ops

image_path = "/mnt/disk0/dat/pascal2010/dat/VOCtrainval/VOC2010"
train_test_path = "/mnt/disk0/dat/pascal2010/Annotation/pascal_person_part/pascal_person_part_trainval_list/" 

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

'''
im_list: contains paths to the person images. shape:(1716,)
gt_list: contains paths to ground truth images. shape:(1716,)
The images order in two lists coresponds to each other. 
'''

im_list = []
gt_list = []

j = 0
with open(train_test_path+'train.txt') as f:
    for line in f:
        im, gt = line.split(" ")
        im_list.append(im)
        gt_list.append(gt.strip())

config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=config) as sess:
    
    vgg_fcn = fcn8_vgg.FCN8VGG()

    image_filename_placeholder = tf.placeholder(tf.string)
    annotation_filename_placeholder = tf.placeholder(tf.string)
    image_tensor = tf.read_file(image_filename_placeholder)
    annotation_tensor = tf.read_file(annotation_filename_placeholder)
    img = tf.image.decode_jpeg(image_tensor, channels=3)
    gt = tf.image.decode_png(annotation_tensor, channels=1)

    bg_label_tensor = tf.to_float(tf.equal(gt, 0, name="Background"),name="bg_float")
    hd_label_tensor = tf.to_float(tf.equal(gt, 38, name="Head"),name="hd_float")
    ts_label_tensor = tf.to_float(tf.equal(gt, 75, name="Tosor"),name="ts_float")
    ua_label_tensor = tf.to_float(tf.equal(gt, 113, name="Upper_Arm"),name="ua_float")
    la_label_tensor = tf.to_float(tf.equal(gt, 14, name="lower_Arm"),name="la_float")
    ul_label_tensor = tf.to_float(tf.equal(gt, 52, name="Upper_Leg"),name="ul_float")
    ll_label_tensor = tf.to_float(tf.equal(gt, 89, name="Lower_Leg"),name="ll_float")
    combined_mark = tf.concat(axis=2, values=[bg_label_tensor,hd_label_tensor,
                                             ts_label_tensor, ua_label_tensor,
                                             la_label_tensor, ul_label_tensor,
                                             ll_label_tensor])
    flat_labels = tf.reshape(tensor=combined_mark, shape=(-1, 7))

    img_float = tf.to_float(img)
    img_batch = tf.expand_dims(img_float, axis=0)
    vgg_fcn.build(img_batch, train=True, num_classes=7,debug=True,random_init_fc8=True)
    predition = vgg_fcn.upscore32

    flat_logits = tf.reshape(tensor=predition, shape=(-1, 7))
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                              labels=flat_labels)
    with tf.name_scope("loss"):
        loss_mean = tf.reduce_mean(cross_entropies)
        loss_op = tf.summary.scalar("loss",loss_mean)

    with tf.variable_scope("adam_vars"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-6)
        train_step = optimizer.minimize(loss_mean)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    write = tf.summary.FileWriter("log/lr"+str(1e-6), sess.graph)

    try:
        for epoch in range(1):
            for i in range(0,10):
                img_filename = image_path+im_list[i]
                gt_filename = image_path+gt_list[i]
                feed_dictionary = {image_filename_placeholder:img_filename,
                                   annotation_filename_placeholder:gt_filename}
                sess.run(train_step, feed_dict=feed_dictionary)
                j+=1
                if j % 2 == 0:
                    temp_loss,loss_summary = sess.run([loss_mean,loss_op], feed_dict=feed_dictionary)
                    write.add_summary(loss_summary, global_step=j)
                    print("Temporay loss: %f,step is %d" % (temp_loss,j))
                    print("The time is %s" % str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        sys.exit()

    except (KeyboardInterrupt, SystemExit):

        filters = ['conv1_1/filter:0','conv1_1/biases:0',"conv1_2/filter:0","conv1_2/biases:0",
                "conv2_1/filter:0","conv2_1/biases:0","conv2_2/filter:0","conv2_2/biases:0",
                "conv3_1/filter:0","conv3_1/biases:0","conv3_2/filter:0","conv3_2/biases:0",
                "conv3_3/filter:0","conv3_3/biases:0",
                "conv4_1/filter:0","conv4_1/biases:0","conv4_2/filter:0","conv4_2/biases:0",
                "conv4_3/filter:0","conv4_3/biases:0","conv5_1/filter:0","conv5_1/biases:0",
                "conv5_2/filter:0","conv5_2/biases:0","conv5_3/filter:0","conv5_3/biases:0",
                "fc6/weights:0","fc6/biases:0","fc7/weights:0","fc7/biases:0","score_fr/weights:0",
               "score_fr/biases:0", "upscore2/up_filter:0","score_pool4/weights:0",
                "score_pool4/biases:0","upscore4/up_filter:0","score_pool3/weights:0",
                "score_pool3/biases:0","upscore32/up_filter:0"]

        c11w,c11b,c12w,c12b,c21w,c21b,c22w,c22b,c31w,c31b,c32w,c32b,c33w,c33b,c41w,c41b,c42w,c42b,c43w,c43b,c51w,c51b,c52w,c52b,c53w,c53b,fc6w,fc6b,fc7w,fc7b,sw,sb,up2w,s4w,s4b,up4w,s3w,s3b,up32 = sess.run(filters)

        print("Saving fine tuned weights")
        save_dict={}
        save_dict={"conv1_1":[c11w,c11b], "conv1_2":[c12w,c12b],"conv2_1":[c21w,c21b],"conv2_2":[c22w,c22b],
                "conv3_1":[c31w,c31b],"conv3_2":[c32w,c32b],"conv3_3":[c33w,c33b],"conv4_1":[c41w,c41b],"conv4_2":[c42w,c42b],
                "conv4_3":[c43w,c43b],"conv5_1":[c51w,c51b],"conv5_2":[c52w,c52b],"conv5_3":[c53w,c53b],
                "fc6":[fc6w,fc6b],"fc7":[fc7w,fc7b], "score_fr":[sw,sb], "upscore2":[up2w],
                "score_pool4":[s4w,s4b],"upscore4":[up4w],"score_pool3":[s3w,s3b],"upscore32":[up32]}
        np.save("tune_fcn8s_weights_test.npy", save_dict)
        print("Fine-tuned weight saved")
        raise
