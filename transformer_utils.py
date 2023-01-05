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
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


class EarlyStopping:
	def __init__(self, patience=1, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_eval_loss = np.inf

	def check(self, eval_loss):
		if eval_loss < self.min_eval_loss:
			self.min_eval_loss = eval_loss
			self.counter = 0
		elif eval_loss > (self.min_eval_loss + self.min_delta):
			self.counter += 1
			print(f'Early stopping counter: {self.counter}/{self.patience}')

			if self.counter >= self.patience:
				return True

		return False

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