import torch
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm
from transformer_utils import get_logger

logger = get_logger(__name__)


class TokenizedSample():
	def __init__(self, guid, src_ids, src_mask, tgt_ids, tgt_mask):
		self.guid = guid
		self.src_ids = src_ids
		self.src_mask = src_mask
		self.tgt_ids = tgt_ids
		self.tgt_mask = tgt_mask


class DataProcessor():
	def create_dataset(self, tokenized_samples):
		all_guid = torch.tensor([s.guid for s in tokenized_samples], dtype=torch.long)
		all_src_ids = torch.tensor([s.src_ids for s in tokenized_samples], dtype=torch.long)
		all_src_mask = torch.tensor([s.src_mask for s in tokenized_samples], dtype=torch.long)
		all_tgt_ids = torch.tensor([s.tgt_ids for s in tokenized_samples], dtype=torch.long)
		all_tgt_mask = torch.tensor([s.tgt_mask for s in tokenized_samples], dtype=torch.long)

		dataset = TensorDataset(all_guid, all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask)
		return dataset

	def create_dataloader(self, dataset, batch_size):
		data_sampler = RandomSampler(dataset)
		dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size, drop_last=True)
		return dataloader

	def create_distributed_dataloader(self, rank, world_size, dataset, batch_size):
		data_sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
		dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size, drop_last=True, shuffle=False)
		return dataloader
		

class CSVProcessor(DataProcessor):
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def get_tokenized_samples(self, csv_file, max_src_len, max_tgt_len, max_rows=None):
		logger.info(f'*** Reading {csv_file} ***')
		df = pd.read_csv(csv_file, sep='\t', encoding='utf8', index_col=None, nrows=max_rows)

		logger.info(f'*** 5 first rows ***')
		print(df.head())
		logger.info(f'*** 5 last rows ***')
		print(df.tail())

		samples = []

		for index, row in tqdm(df.iterrows(), total=len(df), desc='Rows'):
			guid = row['guid']
			src = row['article']
			tgt = row['abstract']
			tokenized_src = self.tokenizer(src, max_length=max_src_len, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='np')
			tokenized_tgt = self.tokenizer(tgt, max_length=max_tgt_len, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='np')

			sample = TokenizedSample(
				guid=guid,
				src_ids=tokenized_src.input_ids[0],
				src_mask=tokenized_src.attention_mask[0],
				tgt_ids=tokenized_tgt.input_ids[0],
				tgt_mask=tokenized_tgt.attention_mask[0]
			)

			samples.append(sample)

		return samples
