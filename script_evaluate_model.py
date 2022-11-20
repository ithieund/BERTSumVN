import os
import sys
import random
import numpy as np
import torch
import json
import torch.nn.functional as F
from datetime import datetime
from transformer_preprocess import DataProcessor
from transformers import AutoTokenizer
from transformer_model import BertAbsSum
from tqdm import tqdm
from transformer_utils import *
from rouge_score import rouge_scorer
from params_helper import Params, Constants

# Setup logger
logger = get_logger(__name__)

BOS_TOKEN = Constants.BOS_TOKEN
EOS_TOKEN = Constants.EOS_TOKEN
SEED = 0

# Set visible GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus

# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(Params.bert_model)


def set_seed(seed):
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	else:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)


def load_data():
	logger.info('*** Loading test dataset ***')
	processor = DataProcessor()

	dataset = torch.load(Params.data_path)
	dataloader = processor.create_dataloader(dataset, Params.batch_size)

	logger.info('*** Checking data ***')
	batch = next(iter(dataloader)) # Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
	batch_guids = batch[0]
	batch_src_ids = batch[1]
	batch_tgt_ids = batch[3]

	for i in range(len(batch_guids)):
		logger.info(f'Sample {batch_guids[i]}')
		logger.info(f'Source: {tokenizer.decode(batch_src_ids[i], skip_special_tokens=True)}')
		logger.info(f'Target: {tokenizer.decode(batch_tgt_ids[i], skip_special_tokens=True)}')

	return dataloader


def load_model(device):
	# Load config
	logger.info(f'*** Loading config from {Params.config_path} ***')

	with open(Params.config_path, 'r') as f:
		config = json.load(f)

	# Load model
	config['freeze_encoder'] = Params.freeze_encoder
	model = BertAbsSum(bert_model_path=Params.bert_model, config=config, device=device)
	
	logger.info(f'*** Loading model state dict: {Params.model_path} ***')
	checkpoint = torch.load(Params.model_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])

	return model.to(device)


def greed_decode_one_sample(model, dataloader, device):
	print('\n\n')
	logger.info('***** Greedy decode one sample *****')
	model.eval()
	batch = next(iter(dataloader)) # Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
	batch_src_ids = batch[1]
	batch_tgt_ids = batch[3]

	pred = model.greedy_decode(batch)
	
	logger.info(f'Source: {tokenizer.decode(batch_src_ids[0], skip_special_tokens=True)}')
	logger.info(f'Target: {tokenizer.decode(batch_tgt_ids[0], skip_special_tokens=True)}')
	logger.info(f'Greedy Generated: {tokenizer.decode(pred[0].cpu().numpy(), skip_special_tokens=True)}')


def decode_one_sample(model, dataloader, device):
	print('\n\n')
	logger.info('***** Evaluating one sample *****')
	model.eval()
	batch = next(iter(dataloader)) # Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
	batch_guids = batch[0]
	batch_src_ids = batch[1]
	batch_src_mask = batch[2]
	batch_tgt_ids = batch[3]
	batch_tgt_mask = batch[4]

	pred = model.beam_decode(
		batch_guids=batch_guids[0].unsqueeze(0),
		batch_src_seq=batch_src_ids[0].unsqueeze(0),
		batch_src_mask=batch_src_mask[0].unsqueeze(0),
		beam_size=Params.beam_size,
		n_best=Params.n_best)

	logger.info(f'Source: {tokenizer.decode(batch_src_ids[0]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]}')
	logger.info(f'Target=======================: {tokenizer.decode(batch_tgt_ids[0]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]}')

	beam_hypotheses = pred[0]

	for h in range(len(beam_hypotheses)):
		score, tokens = beam_hypotheses[h]
		logger.info(f'Beam Search H{h + 1} (score={"{:.3f}".format(-score)}): {tokenizer.decode(tokens).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]}')


def calculate_rouge(targets, summaries):
	scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
	scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
	avg_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

	for target, summary in zip(targets, summaries):
		score = scorer.score(target, summary)
		rouge1 = score['rouge1'].fmeasure
		rouge2 = score['rouge2'].fmeasure
		rougel = score['rougeL'].fmeasure

		scores['rouge1'].append(rouge1)
		scores['rouge2'].append(rouge2)
		scores['rougeL'].append(rougel)

	avg_scores['rouge1'] = np.average(scores['rouge1'])
	avg_scores['rouge2'] = np.average(scores['rouge2'])
	avg_scores['rougeL'] = np.average(scores['rougeL'])

	return avg_scores


def do_evaluate(model, dataloader, device):
	set_seed(SEED)

	if Params.quick_test:
		# greed_decode_one_sample(model, dataloader, device)
		decode_one_sample(model, dataloader, device)

	print('\n\n')
	logger.info("***** Running evaluation *****")
	model.eval()

	eval_log = {'arguments': vars(Params), 'decoding': [], 'rouge_score': {}}
	ref_list = []
	sum_list = []

	# Decode samples
	with torch.no_grad():
		batch_count = 0

		for batch in tqdm(dataloader, desc='Evaluation step', position=0):
			# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
			batch_guids = batch[0]
			batch_src_ids = batch[1]
			batch_src_mask = batch[2]
			batch_tgt_ids = batch[3]
			batch_tgt_mask = batch[4]

			pred = model.beam_decode(
				batch_guids=batch_guids,
				batch_src_seq=batch_src_ids,
				batch_src_mask=batch_src_mask,
				beam_size=Params.beam_size,
				n_best=Params.n_best)

			for i in range(len(batch_guids)):
				tgt_seq = tokenizer.decode(batch_tgt_ids[i]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]
				pred_seq = tokenizer.decode(pred[i][0][1]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]

				eval_log['decoding'].append({
					'guid': batch_guids[i].item(),
					'target': tgt_seq,
					'beam_decode': {
						'h1': tokenizer.decode(pred[i][0][1]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0],
						'h2': tokenizer.decode(pred[i][1][1]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0],
						'h3': tokenizer.decode(pred[i][2][1]).split(BOS_TOKEN)[1].split(EOS_TOKEN)[0]
					}
				})

				ref_list.append(tgt_seq)
				sum_list.append(pred_seq)

			batch_count += 1

			if Params.quick_test and batch_count == 5:
				break

	rouge_score = calculate_rouge(ref_list, sum_list)
	eval_log['rouge_score'] = rouge_score

	# Save output file
	normalized_model_name = Params.bert_model.replace('/', '_')
	output_file_path = f'eval_log_{normalized_model_name}_{datetime.now().strftime("%Y.%m.%d_%H.%M.%S")}.json'

	with open(os.path.join(Params.output_dir, output_file_path), 'w', encoding='utf8') as f:
		json.dump(eval_log, f, indent=4, ensure_ascii=False)

	rouge1 = rouge_score['rouge1']
	rouge2 = rouge_score['rouge2']
	rougeL = rouge_score['rougeL']

	logger.info('*** Evaluation results ***')
	logger.info(f'Rouge-1: {rouge1}')
	logger.info(f'Rouge-2: {rouge2}')
	logger.info(f'Rouge-L: {rougeL}')
	logger.info('Evaluation finished.')


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info(f'Using device: {device}')
	
	model = load_model(device)
	dataloader = load_data()

	do_evaluate(model, dataloader, device=device)
	logger.info('Finished.')
	sys.exit(0)