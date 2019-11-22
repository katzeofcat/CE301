#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
pickle_in = open("x_tr.pickle","rb")# input image data
x = pickle.load(pickle_in)

pickle_in = open("y_tr.pickle","rb")# input label data
lables_tr= pickle.load(pickle_in)
#normalization
data_tr=x/255.0

md = keras.Sequential([
	#covlnet input 32x32x3
	tf.keras.layers.Conv2D(
		filters=32, 
		kernel_size=[3,3], 
		padding='same', 
		activation='relu', 
		input_shape=data_tr.shape[1:]),
	#pooling input(32x32x32)
	tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
	#covlnet2 input (16x16x32)
	tf.keras.layers.Conv2D(
		filters=64, 
		kernel_size=[3,3], 
		padding='same', 
		activation='relu'),
	#pooling input(64x16x16)
	tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

	#FC
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation='sigmoid')])
md.summary()
md.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
callbacks = [keras.callbacks.TensorBoard(log_dir='/home/edgar/test/tb')]
md.fit(data_tr,lables_tr,
	batch_size=100,
	epochs=50,
	callbacks=callbacks)
md.save('/home/edgar/test/md.h5')
frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in md.outputs])
tf.train.write_graph(frozen_graph, '/home/edgar/test/', 'logs.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, '/home/edgar/test/', 'logs.pb', as_text=False)
