import tensorflow as tf
import typing as t
import reader as r
from concurrent.futures import ProcessPoolExecutor

import numpy as np

IMAGE_SHAPE = (255, 255)
p = ProcessPoolExecutor()


# Combine a series of events into a tensor
def collapse(series: t.Iterator[r.Event]) -> tf.Tensor:
	# Sort events by time.
	collected = list(series)
	collected.sort(key=lambda e: e.timestamp)

	# Create template image.
	#image = tf.zeros(IMAGE_SHAPE, dtype=tf.uint8)
	image = np.zeros(IMAGE_SHAPE, dtype="uint8")
	# Apply the changes to the tensor.
	for event in collected:
		image[event.x, event.y] = event.polarity

	# Return result.
	return tf.constant(image)


def collate(hyper_series: t.List[t.Iterator[r.Event]]) -> t.Iterator[tf.Tensor]:
	images = p.map(collapse, hyper_series, chunksize=64)
	#images = map(lambda elem: elem.get(), peter)
	#images = map(collapse, hyper_series)
	return images


# The badger batches series.
def badger(input: r.MainStruct) -> t.Dict[int, t.Iterator[tf.Tensor]]:
	result = {}
	for n, hyper_series in input.items():
		result[n] = collate(hyper_series)
	return result