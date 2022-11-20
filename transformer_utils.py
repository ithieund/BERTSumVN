from typing import Any
import torch
import torch.nn as nn
import re
import random
import numpy as np
import logging
from functools import wraps
from time import time
from params_helper import Constants, Params


def get_logger(logger_name):
	logging.basicConfig(
		format='%(asctime)s - %(levelname)s - %(name)s:: %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
		level=logging.INFO)

	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.INFO)

	if Params.logger_debug == True:
		logger.setLevel(logging.DEBUG)

	return logger


def timing(f):
	@wraps(f)
	def wrap(*args, **kwargs):
		ts = time()
		result = f(*args, **kwargs)
		te = time()

		if Params.logger_debug == True:
			print('===>>> Class: %r, Function: %r, Running time: %2.4f sec <<<===' % (f.__module__, f.__name__, te - ts))

		return result

	return wrap


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

class MyDataParallel(nn.DataParallel):
	def _forward_unimplemented(self, *input: Any) -> None:
		pass

	def __getattr__(self, name):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)


def init_parallel_model(model):
	if torch.cuda.device_count() > 1:
		print('Parallel training using', torch.cuda.device_count(), 'GPUs!')
		model = MyDataParallel(model)

	return model


def print_sample(sample, tokenizer):
	print('*** Sample ***')
	print('- guid: %s' % (sample.guid))
	print('- src ids: %s' % ' '.join([str(x) for x in sample.src_ids]))
	print('- src tokens: %s' % ' '.join(tokenizer.convert_ids_to_tokens(sample.src_ids)))
	print('- src mask: %s' % ' '.join([str(x) for x in sample.src_mask]))
	print('- tgt ids: %s' % ' '.join([str(x) for x in sample.tgt_ids]))
	print('- tgt tokens: %s' % ' '.join(tokenizer.convert_ids_to_tokens(sample.tgt_ids)))
	print('- tgt mask: %s' % ' '.join([str(x) for x in sample.tgt_mask]))


def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if isinstance(text, str):
		return text
	elif isinstance(text, bytes):
		return text.decode("utf-8", "ignore")
	else:
		raise ValueError("Unsupported string type: %s" % (type(text)))


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rouge(hyp, ref, n):
	scores = []

	for h, r in zip(hyp, ref):
		r = re.sub(r'[UNK]', '', r)
		r = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', r)
		r = re.sub(r'\d', '', r)
		r = re.sub(r'[a-zA-Z]', '', r)
		count = 0
		match = 0

		for i in range(len(r) - n):
			gram = r[i:i + n]
			if gram in h:
				match += 1
			count += 1

		scores.append(0 if count == 0 else match / count)

	return np.average(scores)

# TODO: convert to pytorch
def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

if __name__ == "__main__":
	hyp = ['Trường_học đang đóng cửa vì dịch covid', 'Bệnh_viện đang quá tải vì dịch covid']
	ref = ['Trường_học đang mở cửa chào năm học mới', 'Bệnh_viện đang đóng cửa vì dịch covid']
	print(rouge(hyp, ref, 1))
	print(rouge(hyp, ref, 2))
