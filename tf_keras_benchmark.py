import numpy as np
import time
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
from tensorflow.python.keras.layers import Lambda

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_string("gpu_id", "0", "idx of GPU using")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("image_size", 224, "Image size")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

# generate synthetic data
x = np.random.randint(0, 1, size=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))
x = x.astype("float32")
y = np.random.randint(0, 1000, size=FLAGS.batch_size)
y = tf.keras.utils.to_categorical(y, 1000)

# make dataset
# dataset = tf.data.Dataset.from_tensor_slices((x, y))
# dataset = dataset.batch(FLAGS.batch_size)
# dataset = dataset.repeat()

# # def tf.data.Dataset
features_placeholder = tf.keras.layers.Input(shape=(FLAGS.image_size, FLAGS.image_size, 3))
labels_placeholder = tf.placeholder(y.dtype, y.shape)

# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# dataset = dataset.batch(FLAGS.batch_size).filter(lambda features, labels: tf.equal(tf.shape(labels)[0], FLAGS.batch_size))
# dataset = dataset.repeat(500)
# iterator = dataset.make_initializable_iterator()
# inputs, labels = iterator.get_next()

# build model
model = tf.keras.applications.resnet50.ResNet50(input_shape=(FLAGS.image_size, FLAGS.image_size, 3), weights=None)

# build slim model
# with slim.arg_scope(slimNet.resnet_utils.resnet_arg_scope(batch_norm_decay=0.99)):
#     _, layers_dict = slimNet.resnet_v2.resnet_v2_50(features_placeholder, num_classes=1000, global_pool=True, is_training=True)
#     logits = layers_dict['resnet_v2_50/logits']
#     logits = Lambda(tf.keras.layers.Flatten())(logits)
#     outputs = tf.keras.layers.Dense(1000, activation=tf.nn.softmax)(logits)
    
# model = tf.keras.Model(inputs=features_placeholder, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())

# Create training op.
t = time.time()
model.fit(x, y, epochs=10, steps_per_epoch=100)
print("Elapsed time = ", time.time() - t)