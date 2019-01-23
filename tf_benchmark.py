import numpy as np
import time
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_string("gpu_id", "0", "idx of GPU using")
tf.app.flags.DEFINE_string("model", "resnet50", "select from resnet50, googlenet")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.app.flags.DEFINE_integer("image_size", 224, "Image size")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

# generate synthetic data
x = np.random.randint(0, 1, size=(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3))
x = x.astype("float32")
y = np.random.randint(0, 1000, size=FLAGS.batch_size)
y = tf.keras.utils.to_categorical(y, 1000)

# def tf.data.Dataset
features_placeholder = tf.placeholder(x.dtype, x.shape)
labels_placeholder = tf.placeholder(y.dtype, y.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
dataset = dataset.batch(FLAGS.batch_size).filter(lambda features, labels: tf.equal(tf.shape(labels)[0], FLAGS.batch_size))
dataset = dataset.repeat(500)
iterator = dataset.make_initializable_iterator()
inputs, labels = iterator.get_next()

# build model
if FLAGS.model == "slim":
    print("Slim model")
    with slim.arg_scope(slimNet.resnet_utils.resnet_arg_scope(batch_norm_decay=0.99)):
        _, layers_dict = slimNet.resnet_v2.resnet_v2_50(inputs, num_classes=1000, global_pool=True, is_training=True)
        logits = layers_dict['resnet_v2_50/logits']
        logits = tf.keras.layers.Flatten()(logits)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                       logits=logits)       
else:
    keras_model = tf.keras.applications.resnet50.ResNet50(input_shape=(FLAGS.image_size, FLAGS.image_size, 3), weights=None)
    output = keras_model(inputs)
    loss = tf.losses.log_loss(labels, output)
    print("Keras model")


# Create training op.
with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step=tf.train.get_global_step())

res = []
# Start session and training
with tf.train.MonitoredTrainingSession() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: x,
                                          labels_placeholder: y})
    print("RUNNING WARMUP")
    for w in range(5):
        sess.run(train_step)
    print("WARMUP DONE")
    for b in range(1, 61):
        t = time.time()
        sess.run(train_step)
        t1 = time.time()
        _loss = sess.run(loss)
        if b % 10 == 0:
            print("Num:", b, ", Loss: ", _loss, ", Elapsed time: ", t1 - t, "Images/sec: ", (FLAGS.batch_size / (t1-t)))
        res.append(FLAGS.batch_size / (t1-t))
print(np.mean(res), " +- ", np.std(res))
