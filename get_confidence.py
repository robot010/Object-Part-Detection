import numpy as np
import os
import tuned_fcn8_vgg
import tensorflow as tf
import skimage.segmentation as ss
import scipy as scp

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

images_path = "/mnt/disk0/dat/pascal2010/dat/VOCtrainval/VOC2010/"
train_test_path = "/mnt/disk0/dat/pascal2010/Annotation/pascal_person_part/pascal_person_part_trainval_list/"
save_path = "/mnt/disk0/mid/glstm/confidence_map/"

im_list = []
gt_list = []
with open(train_test_path+'train.txt') as f:
    for line in f:
        im, gt = line.split(" ")
        im_list.append(im)
        gt_list.append(gt.strip())

with tf.Session() as sess:
    image_placeholder = tf.placeholder("float")
    batch_images = tf.expand_dims(image_placeholder,0)
    vgg_fcn = tuned_fcn8_vgg.tuned_FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True, num_classes=7)
    print("Finished building network")

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(0,1716):

        image_name = im_list[i]
        image_path = images_path + image_name
        image = scp.misc.imread(image_path)
        feed_dict = {image_placeholder: image}
        
        input_tensor = vgg_fcn.upscore32
        confidence_map = sess.run(input_tensor, feed_dict=feed_dict)
        
        save_name = image_name[12:-4]
        np.save(save_path+save_name, confidence_map)
        print("Finish image %d" % i) 
