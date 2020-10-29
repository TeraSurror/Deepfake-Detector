from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

########################################################
# Imports for Image model
########################################################
import numpy as np
from keras.preprocessing import image
import pandas as pd
from keras_facenet import FaceNet
from tqdm import tqdm_notebook
#from sklearn.externals import joblib 
import pickle 

## For Face extraction
from mtcnn import MTCNN
import cv2

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import argparse

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam, Nadam
from keras.applications.xception import Xception
# from keras.applications.resnet_v2 import ResNet50V2
from keras import backend as K
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers.pooling import MaxPooling2D

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 

from sklearn.linear_model import SGDClassifier
########################################################


def create_base_network(embedding_size):
	"""
	Base network to be shared (eq. to feature extraction).
	"""
	main_input = Input(shape=(512, ))
	x = Dense(256, activation='relu', kernel_initializer='he_uniform')(main_input)
	x = Dropout(0.1)(x)
	x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	y = Dense(embedding_size)(x)
	base_network = Model(main_input, y)

	return base_network

def extract_face_from_image(image):
  detector = MTCNN()  

  faces_image_data = detector.detect_faces(image)

  faces_image_data_op = []

  for face_image_data in faces_image_data: 
  
    x = face_image_data["box"][0]
    y = face_image_data["box"][1]
    w = face_image_data["box"][2]
    h = face_image_data["box"][3]

    face_image = image[y:y+h, x:x+w]

    face_image_dict = {
        "x" : x,
        "y" : y,
        "w" : w,
        "h" : h,
        "face_image" : face_image
    }

    faces_image_data_op.append(face_image_dict)

  return faces_image_data_op


def extract_facenet_embeddings(face_image):
  embedder = FaceNet()
  x = image.img_to_array(face_image)
  x = np.expand_dims(x, axis=0)
  embeddings = embedder.embeddings(x)

  return embeddings


def predict_deepfake(input_embedding, input_image_shape, embedding_size):

  testing_embeddings = create_base_network(input_image_shape, embedding_size=embedding_size)

  # Load Pretrained model
  model_tr = load_model("triplets_semi_hard.hdf5", custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})
  
  # Grabbing the weights from the trained network
  for layer_target, layer_source in zip(testing_embeddings.layers, model_tr.layers[2].layers):
    weights = layer_source.get_weights()
    layer_target.set_weights(weights)
    del weights        

  prediction = testing_embeddings.predict(input_embedding)

  
  sgd_loaded = pickle.load('sgd.pkl')
  pred = sgd_loaded.predict(prediction)

  return pred


def detect_deepfake_image(img_path):

    print(img_path)
    # Read image from path

    img = cv2.imread(img_path)
    print(img)


    # Extract face from image
    faces_image_data = extract_face_from_image(img)

    for face_image_data in faces_image_data:
        x = face_image_data["x"]
        y = face_image_data["y"]
        w = face_image_data["w"]
        h = face_image_data["h"]

        face_image = face_image_data["face_image"]

        # Extract FaceNet Embeddings
        face_embeddings = extract_facenet_embeddings(face_image)

        # Reshape embeddings into 512 dimension vector (sort of)
        x1 = np.reshape(face_embeddings, (1 ,512))

        # Predict
        output = predict_deepfake(x1, (None, 512), 64)
        print(output)



        label = "Real"

        if output[0] == 1:
            label = "Fake"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
        cv2.putText(img, label, (x+5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imwrite('model/output/op.jpg', img)

########################################################

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    
    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance

  
    
    
########################################################
# Create your views here.
def index(request):

    context = {'a' : 1}

    return render(request, 'index.html', context=context)

def detect_deepfake(request):

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        print(myfile.name)
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        detect_deepfake_image("D:/harsh/Projects/Django Projects/DeepFakeWebApp/media/" + filename)

        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url,
            'output_file_url' : 'DeepFakeWebApp/media/output/op.jpg'
        })
    return render(request, 'index.html')