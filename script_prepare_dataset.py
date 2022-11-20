import os
import sys
import torch
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from transformer_preprocess import CSVProcessor
from transformer_utils import *
from params_helper import Params, Constants

# Setup logger
logger = get_logger(__name__)


def prepare_data(tokenizer, csv_file_path, data_type, max_rows, max_src_len, max_tgt_len):
	print('\n\n')
	logger.info(f'Preparing {data_type} data...')
	preprocessed_dataset_path = os.path.splitext(csv_file_path)[0] + ('' if max_rows < 0 else f'_{max_rows}rows') + f'_{max_src_len}tokens.pt'
	processor = CSVProcessor(tokenizer=tokenizer)

	logger.info(f'Loading {data_type} samples...')
	samples = processor.get_tokenized_samples(
		csv_file=csv_file_path,
		max_src_len=max_src_len,
		max_tgt_len=max_tgt_len,
		max_rows=None if max_rows < 0 else max_rows)
	print_sample(sample=samples[0], tokenizer=tokenizer)

	logger.info(f'Building {data_type} dataset...')
	dataset = processor.create_dataset(samples)

	logger.info(f'Saving {data_type} dataset into {preprocessed_dataset_path}')
	torch.save(dataset, preprocessed_dataset_path)


# Usage: python prepare_dataset.py -visible_gpus='2' -bert_model='vinai/phobert-large' -max_rows=-1 -max_src_len=512 -max_tgt_len=128 -train_csv_path='path/to/file' -valid_csv_path='path/to/file' -test_csv_path='path/to/file'
if __name__ == '__main__':
	# Validate csv files
	if not os.path.exists(Params.train_csv_path):
		logger.info(f'Error: Train data file {Params.train_csv_path} does not exist!')
		sys.exit(0)

	if not os.path.exists(Params.valid_csv_path):
		logger.info(f'Error: Valid data file {Params.valid_csv_path} does not exist!')
		sys.exit(0)

	if not os.path.exists(Params.test_csv_path):
		logger.info(f'Error: Test data file {Params.test_csv_path} does not exist!')
		sys.exit(0)

	# Set visible GPUs
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus

	# Init tokenizer
	tokenizer = AutoTokenizer.from_pretrained(Params.bert_model, local_files_only=False)

	# Process data
	prepare_data(
		tokenizer=tokenizer,
		csv_file_path=Params.train_csv_path,
		data_type='train',
		max_rows=Params.max_rows,
		max_src_len=Params.max_src_len,
		max_tgt_len=Params.max_tgt_len,
	)

	prepare_data(
		tokenizer=tokenizer,
		csv_file_path=Params.valid_csv_path,
		data_type='valid',
		max_rows=Params.max_rows,
		max_src_len=Params.max_src_len,
		max_tgt_len=Params.max_tgt_len,
	)

	prepare_data(
		tokenizer=tokenizer,
		csv_file_path=Params.test_csv_path,
		data_type='test',
		max_rows=Params.max_rows,
		max_src_len=Params.max_src_len,
		max_tgt_len=Params.max_tgt_len,
	)

	logger.info(f'Finished.')
	sys.exit(0)