import tensorflow as tf
import datetime
from badger import badger, IMAGE_SHAPE, p
import typing as t
from reader import folder_to_dict
import pathlib as pl

log_dir = "logs/fit-nmnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
TRAIN_DIR = "N-MNIST/Train"
TEST_DIR = "N-MNIST/Test"

TRAIN_STORE = "ptrainp"
TEST_STORE = "ptestp"


def dict_to_dataset(
        input: t.Dict[int, t.Iterator[tf.Tensor]]) -> tf.data.Dataset:
	def gen():
		for n, images in input.items():
			for image in images:
				yield (image, n)

	typespec = (tf.TensorSpec(shape=IMAGE_SHAPE, dtype=tf.uint8),
	            tf.TensorSpec(shape=(), dtype=tf.uint8))
	return tf.data.Dataset.from_generator(gen, output_signature=typespec)


def normalize_img(image, label):
	"""Normalizes images: `uint8` -> `float32`."""
	return tf.cast(image, tf.float32) / 255., label


if not (pl.Path(TRAIN_STORE).exists() and pl.Path(TEST_STORE).exists()):

	print("Loading, badgering")
	train = badger(folder_to_dict(TRAIN_DIR))
	test = badger(folder_to_dict(TEST_DIR))

	print("Datasetting")
	ds_train = dict_to_dataset(train)
	ds_test = dict_to_dataset(test)

	ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
	ds_train = ds_train.batch(128, num_parallel_calls=tf.data.AUTOTUNE)
	ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

	ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
	ds_test = ds_test.batch(128, num_parallel_calls=tf.data.AUTOTUNE)
	ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

	print("Saving training set")
	tf.data.experimental.save(ds_train, TRAIN_STORE)

	print("Saving testing set")
	tf.data.experimental.save(ds_test, TEST_STORE)

	print("Saved")
else:
	ds_train = tf.data.experimental.load(TRAIN_STORE)
	ds_test = tf.data.experimental.load(TEST_STORE)

p.shutdown()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adadelta(10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(ds_train,
          epochs=30,
          validation_data=ds_test,
          callbacks=[tensorboard_callback])

model.save("model-nmnist")