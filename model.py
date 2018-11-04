import data_utils
import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras import layers, models, layers, optimizers, callbacks, regularizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.data import Dataset, Iterator

TRAIN_IMG_DIR = 'spectrogram_images/train/'
TEST_IMG_DIR = 'spectrogram_images/test/'
VAL_IMG_DIR = 'spectrogram_images/val/'

IMG_HEIGHT = 500
IMG_WIDTH = 500

NUM_EPOCHS = 15
BATCH_SIZE = 64
NUM_GPUS = 4

NUM_CLASSES = 7

L2_LAMBDA = 0.001
LEARNING_RATE = 1e-5

def generate_model():
    conv_base = tf.contrib.keras.applications.VGG16(include_top = False,
                                                weights = 'imagenet',
                                                input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    conv_base.trainable = True
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, name='dense_1', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(layers.Dropout(rate=0.3, name='dropout_1'))
    model.add(layers.Dense(128, name='dense_2', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(layers.Dropout(rate=0.3, name='dropout_2'))
    model.add(layers.Dense(50, name='dense_3', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(layers.Dropout(rate=0.3, name='dropout_3'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='dense_output'))
    model = multi_gpu_model(model, gpus=NUM_GPUS)
    print(model.summary())
    return model

def generate_labeled_dataset(img_dir, sess):
    features, labels = data_utils.load_data(img_dir)
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, NUM_EPOCHS))
    dataset = dataset.batch(BATCH_SIZE)

    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
    return iterator


def train_and_validate(model, train_dataset, val_dataset):
    optimizer = optimizers.Adam(lr=LEARNING_RATE)
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']

    filepath="saved_models/transfer_learning_epoch_{epoch:02d}_{val_categorical_accuracy:.4f}.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                       monitor='val_categorical_accuracy',
                                       verbose=0,
                                       save_best_only=False)
    callbacks_list = [checkpoint]

    TRAIN_STEPS = 19302 // BATCH_SIZE #TODO change the numbers
    VAL_STEPS = 1927 // BATCH_SIZE

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              steps_per_epoch=TRAIN_STEPS,
              validation_data=val_dataset,
              validation_steps = VAL_STEPS,
              callbacks = callbacks_list)

if __name__=='__main__':
    with tf.Session() as sess:
        model = generate_model()
        train_dataset = generate_labeled_dataset(TRAIN_IMG_DIR, sess)
        val_dataset = generate_labeled_dataset(VAL_IMG_DIR, sess)
        train_and_validate(model, train_dataset, val_dataset)
