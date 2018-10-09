import urllib, os, sys, zipfile
from os.path import dirname
import tfcoreml as tf_converter
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


# Load the TF graph definition
tf_model_path = 'tf_files/retrained_graph.pb'
with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

# For demonstration purpose we show the first 15 ops the TF model
with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
    ops = g.get_operations()
    for i in range(15):
        print('op id {} : op name: {}, op type: "{}"'.format(str(i),ops[i].name, ops[i].type));

# This Inception model uses DecodeJpeg op to read from JPEG images
# encoded as string Tensors. You can visualize it with TensorBoard,
# but we're omitting it here. For deployment we need to remove the
# JPEG decoder and related ops, and replace them with a placeholder
# where we can feed image data in.

# Strip the JPEG decoder and preprocessing part of TF model
# In this model, the actual op that feeds pre-processed image into
# the network is 'Mul'. The op that generates probabilities per
# class is 'softmax/logits'
# To figure out what are inputs/outputs for your own model
# You can use use TensorFlow's summarize_graph or TensorBoard
# Visualization tool for your own models.


input_node_names = ['Mul']
output_node_names = ['final_result']
gdef = strip_unused_lib.strip_unused(
        input_graph_def = original_gdef,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = dtypes.float32.as_datatype_enum)

# Save it to an output file
frozen_model_file = './frozen_model.pb'
with gfile.GFile(frozen_model_file, "wb") as f:
    f.write(gdef.SerializeToString())
