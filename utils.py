import numpy as np
import pdb
from skimage.future import graph
from skimage.measure import regionprops
import tensorflow as tf
import pathos.multiprocessing as mp
import time


def generate_neighbor_matrix(superpixel_map, sequence):
    '''
    This function is to convert the input superpixel image into a 
    matrix with shape (number_of_superpixel, number_of_superpixel).
    It's used to indicate what's the neighbor of a superpixel. 
    "1" --> These two superpixels are neighbour. 
    "0" --> These two superpixels are not neightbour. 
    
    Arg:
        superpixel_map: (Width, Height)
        sequence: Segmentation values that corresponding to labels except background. 
    
    Return:
        Adjacency matrix. 
    '''
    
    pseudo_edge_map = np.ones(np.shape(superpixel_map))
    rag = graph.rag_boundary(superpixel_map, pseudo_edge_map, connectivity=2)
    num_super = rag.number_of_nodes()    
    neighbor_matrix = np.zeros((num_super, num_super))
    
    nodes_list = rag.nodes()
    for i in nodes_list:

        if i not in sequence:
            continue

        for neighbor in rag.neighbors_iter(i):
            neighbor_not_background = np.intersect1d(neighbor, sequence)
            neighbor_matrix[i, neighbor_not_background] = 1
         
    return neighbor_matrix

def get_sequence(superpixel_map, confidence_map):
    """
    This function is to generate a sequence of features based on 
    the confidence_map. The confidence of each superpixel for each 
    label is computed by averaging the confidences of its contained 
    pixels, and the label with highest confidence could be assigned 
    to the superpixel. Note: Only foreground labels are consider.

    Arg:
        superpixel_map: (Width, Height)
        confidence_map: (Width, Height, num_class)

    Return:
        A list containing the sequence. Each element is one of the 
        superpixel's value. 
    """

    info = {}
    id_sequence = []
    seg_values = np.unique(superpixel_map)
    initial_segmentation = np.argmax(confidence_map, axis=2)
    for segVal in seg_values:

        if sum(initial_segmentation[superpixel_map == segVal]) == 0:
            continue
        
        mask = np.zeros((np.shape(superpixel_map)))
        mask[superpixel_map == segVal] = 1
        num_pixel_in_super = np.sum(mask)

        # Compute the class of each superpixel. 
        reshape_mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        super_confidence_map = reshape_mask * confidence_map
        super_class_confidence = np.sum(super_confidence_map, axis=(0,1))/num_pixel_in_super
        cla = np.argmax(super_class_confidence)
        con = super_class_confidence[cla]
        
        # Put all superpixel's info into a dictionary.
        info[segVal] = [cla, con]

    # Generate the sequence based on superpixel's confidence
    sequence = sorted(info.items(), key=lambda x: x[1][1], reverse=True)
    id_sequence = [i[0] for i in sequence]

    return id_sequence
    
def compute_feature_matrix(feature_map, slic_image, sequence):

    flat_feature_map = tf.reshape(feature_map, shape=(-1, 512))
    flat_slic = tf.reshape(slic_image, [-1])
    vals, idx = tf.unique(flat_slic)
    num_seg = tf.shape(vals)[0]
    seg_id = 0
    segment_position = tf.expand_dims(tf.to_float(tf.equal(flat_slic, seg_id)), axis=1)
    initial_means = tf.expand_dims(tf.divide(tf.reduce_sum(tf.multiply(flat_feature_map, segment_position), axis=0),tf.reduce_sum(segment_position)), axis=0)
    count= tf.Variable(tf.constant(0))

    def condition(x):
        count_seg_id = tf.shape(x)[0]
        return tf.less(count_seg_id, num_seg)

    def body(x):
        seg_id = tf.shape(x)[0]
        segment_position = tf.expand_dims(tf.to_float(tf.equal(flat_slic, seg_id)), axis=1)
        segment_means = tf.expand_dims(tf.divide(tf.reduce_sum(tf.multiply(flat_feature_map, segment_position), axis=0),tf.reduce_sum(segment_position)), axis=0)
        #def f1():
        #    global feature_matrixs
        #    feature_matrixs = segment_means 
        #    return feature_matrixs
        r = tf.concat([x, segment_means], 0)
        # r = tf.cond(tf.equal(x, 0), f1, lambda: tf.concat([feature_matrixs, segment_means], 0))
        return r
    
    #x = tf.Variable(tf.constant([0]))
    result = tf.while_loop(condition, body, [initial_means], shape_invariants=[tf.TensorShape([None, 512])])
    #initial_segmentation = tf.argmax(confidence_map, axis=2)
    #flat_seg = tf.reshape(initial_segmentation, [-1])
    #fore_ground = tf.not_equal(flat_seg, [0])
    #fore_ground_idx = tf.where(fore_ground)
    #feature_matrix = tf.segment_mean(flat_feature_map, seg_id)

    return result

#def compute_feature_matrix(slic_map, feature_map_path, sequence):
#
#    feature_map = np.load(feature_map_path)
#    t0 = time.time()
#    print("Start calculating feature matrix")
#
#    def get_feature(superpixel_id):
#        """
#        This function is to return the feature of a superpixel given 
#        the superpixel ID and the feature map. 
#        
#        Arg:
#            feature_map: (Width, Height, num_feature)
#            superpixel_id: int
#            superpixel_map: (Width, Height)
#            
#        return:
#            A vector: (1, num_feature)
#        """
#        mask = np.zeros((np.shape(slic_map)))
#        mask[slic_map == superpixel_id] = 1
#        num_pixel_in_super = np.sum(mask)
#
#        print("feature map of superpixel is coming")
#        super_feature_map = mask.reshape(mask.shape[0],mask.shape[1],1) * feature_map
#        #super_feature_map = mask[:,:,np.newaxis] * feature_map
#        super_feature_vector = np.sum(super_feature_map, axis=(0,1))/num_pixel_in_super
#
#        return super_feature_vector
#
#    #pool = mp.Pool(processes=40)
#    #results = pool.map(get_feature, sequence)
#    results = map(get_feature, sequence)
#    print(time.time() - t0, "Getting the feature matrix")
#
#    return results

def get_seg_gt(gt_mask, seg_id, slic_image):
    """
    This function is to computer the ground truth label of a superpixel, 
    given the superpixel's seg_id. 

    Arg:
        gt_mask: (Width, Height, num_class) Binary
        seg_id: int
        slic_image: (Width, Height)

    return:
        A binary ground truth vector. (1, num_class)
    """
    mask = np.zeros((np.shape(slic_image)))
    mask[slic_image == seg_id] = 1

    reshape_mask = mask.reshape(mask.shape[0],mask.shape[1],1)
    # super_gt_map = mask[:,:,np.newaxis] * gt_mask
    super_gt_map = reshape_mask * gt_mask
    number_of_pixel_reference = 0

    for cla in range(0,7):
        number_of_pixel_in_this_class = np.sum(super_gt_map[:,:,cla])
        if number_of_pixel_in_this_class > number_of_pixel_reference:
            number_of_pixel_reference = number_of_pixel_in_this_class
            eventual_class = cla

    return_vector = np.zeros((1,7))
    return_vector[:,eventual_class] = 1

    return return_vector


def map_back_to_image(feature_matrix, slic_img, sequence):

    """
    This function is to copy each superpixel's Graph 
    LSTM location back to every pixel location of the
    corresponding superpixel. 

    Arg:
        final_hidden: (num_seg, num_feature)
    """
    num_seg = np.shape(feature_matrix)[0]
    num_class = np.shape(feature_matrix)[1]
    output_shape = np.shape(slic_img) + (num_class,)
    output = np.zeros(output_shape)
    for i in range(0,num_seg):
        super_class = feature_matrix[i]
        x_coordinates,y_coordinates = np.where(slic_img==sequence[i])
        for j in range(0,len(x_coordinates)):
            output[x_coordinates[j],y_coordinates[j],:] = super_class
        
    return output



