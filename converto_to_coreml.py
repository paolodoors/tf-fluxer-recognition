import os
import urllib
import tarfile
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


# def download_file_and_unzip(url, dir_path='.'):
#     """Download the frozen TensorFlow model and unzip it.
#     url - The URL address of the frozen file
#     dir_path - local directory
#     """
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     k = url.rfind('/')
#     fname = url[k+1:]
#     fpath = os.path.join(dir_path, fname)

#     if not os.path.exists(fpath):
#         urllib.urlretrieve(url, fpath)
#     tar = tarfile.open(fpath)
#     tar.extractall(dir_path)
#     tar.close()

# inception_v1_url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz'
# download_file_and_unzip(inception_v1_url)



# Strip the JPEG decoder and preprocessing part of TF model
# In this model, the actual op that feeds pre-processed image into 
# the network is 'Mul'. The op that generates probabilities per
# class is 'softmax/logits'

# To figure out what are inputs/outputs for your own model
# You can use use TensorFlow's summarize_graph or TensorBoard
# Visualization tool for your own models.

# Load the TF graph definition
tf_model_path = 'tf_files/retrained_graph.pb'
with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

# # For demonstration purpose we show the first 15 ops the TF model
with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
    ops = g.get_operations()
    for i in range(15):
        print('op id {} : op name: {}, op type: "{}"'.format(str(i),ops[i].name, ops[i].type));

input_node_names = ['contents:0']
output_node_names = ['final_result:0']
gdef = strip_unused_lib.strip_unused(
        input_graph_def = original_gdef,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = dtypes.float32.as_datatype_enum)

# # Save it to an output file
frozen_model_file = './frozen_model.pb'
with gfile.GFile(frozen_model_file, "wb") as f:
    f.write(gdef.SerializeToString())

# Load the TF graph definition
import tensorflow as tf
tf_model_path = './frozen_model.pb'

with open(tf_model_path, 'rb') as f:
    serialized = f.read()
tf.reset_default_graph()
original_gdef = tf.GraphDef()
original_gdef.ParseFromString(serialized)

# Lets get some details about a few ops in the beginning and the end of the graph
with tf.Graph().as_default() as g:
    tf.import_graph_def(original_gdef, name='')
    ops = g.get_operations()
    N = len(ops)
    for i in [0,1,2,N-3,N-2,N-1]:
        print('\n\nop id {} : op type: "{}"'.format(str(i), ops[i].type));
        print('input(s):'),
        for x in ops[i].inputs:
            print("name = {}, shape: {}, ".format(x.name, x.get_shape())),
        print('\noutput(s):'),
        for x in ops[i].outputs:
            print("name = {}, shape: {},".format(x.name, x.get_shape())),


import tfcoreml
# Supply a dictionary of input tensors' name and shape (with batch axis)
input_tensor_shapes = {"input:0":[1,224,224,3]} # batch size is 1

#providing the image_input_names argument converts the input into an image for CoreML
image_input_name = ['input:0']

# Output CoreML model path
coreml_model_file = './sapo.mlmodel'

# The TF model's ouput tensor name
output_tensor_names = ['InceptionV1/Logits/Predictions/Softmax:0']

# class label file: providing this will make a "Classifier" CoreML model
class_labels = 'tf_files/retrained_labels.txt'

# # Call the converter. This may take a while
coreml_model = tfcoreml.convert(
        tf_model_path=tf_model_path,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = image_input_name,
        class_labels = class_labels)