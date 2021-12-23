from typing import Dict, List, Iterator
import numpy as np
import pathlib as pl
from bitstruct import unpack


class Event:
	# Specifies the format. I took it from the readme. If it is wrong, which I
	# think it is because it tells me that the pictures are apparently bigger
	# than 28x28, I am sorry. I could not access the provided code. I wrote an
	# email but did not get an answer in time.
	#
	format = ">u8u8u1u23"

	def __init__(self, binary):
		x, y, p, t = unpack(self.format, binary)
		self.x = x
		self.y = y
		self.polarity = p
		self.timestamp = t


MainStruct = Dict[int, List[Iterator[Event]]]


def reader(main_folder_path: str) -> Dict[int, Iterator[np.ndarray]]:
	folder = pl.Path(main_folder_path)
	datasets = {}
	for subfolder in folder.iterdir():
		if not subfolder.is_dir():
			continue
		files = filter(lambda elem: elem.is_file(), subfolder.iterdir())
		datasets[subfolder.name] = map(
		    lambda file: np.fromfile(file, dtype='uint8'), files)
	return datasets


class BlobRunner():
	def __init__(self, binary_blob):
		self.binary_blob = binary_blob

	def __call__(self, n):
		return Event(self.binary_blob[n:n + 5])


def parse_file(binary_blob: np.ndarray) -> Iterator[Event]:
	r = BlobRunner(binary_blob)
	return map(r, range(0, len(binary_blob), 5))


def parser(data: Dict[int, Iterator[np.ndarray]]) -> MainStruct:
	parsed = {}
	for number, files in data.items():
		example = map(parse_file, files)
		parsed[number] = example

	return parsed


def folder_to_dict(path: str) -> MainStruct:
	raw_data = reader(path)
	return parser(raw_data)