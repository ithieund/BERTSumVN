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
from tqdm.auto import tqdm
from transformer_utils import *
from rouge_score import rouge_scorer
from params_helper import Params, Constants

set_seed(0)

# Setup logger
logger = get_logger(__name__)

BOS_TOKEN = Constants.BOS_TOKEN
EOS_TOKEN = Constants.EOS_TOKEN

# Set visible GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus

# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(Params.bert_model)


def load_data():
	logger.info('*** Loading test dataset ***')
	processor = DataProcessor()

	dataset = torch.load(Params.data_path)
	dataloader = processor.create_dataloader(dataset, Params.batch_size, shuffle=False)

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
	base_dir = os.path.dirname(Params.model_path)
	config_path = os.path.join(base_dir, 'config.json')

	# Load config
	logger.info(f'*** Loading config from {config_path} ***')

	with open(config_path, 'r') as f:
		config = json.load(f)

	# Load model
	model = BertAbsSum(config=config, device=device)
	
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

	logger.info(f'Source: {tokenizer.decode(batch_src_ids[0], skip_special_tokens=True)}')
	logger.info(f'Target=======================: {tokenizer.decode(batch_tgt_ids[0], skip_special_tokens=True)}')

	beam_hypotheses = pred[0]

	for h in range(len(beam_hypotheses)):
		score, tokens = beam_hypotheses[h]
		logger.info(f'Beam Search H{h + 1} (score={"{:.3f}".format(score)}): {tokenizer.decode(tokens, skip_special_tokens=True)}')


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


def save_eval_log(output_dir, training_log):
	checkpoint_name = os.path.splitext(os.path.basename(Params.model_path))[0]
	log_file_name = f'eval_log_{checkpoint_name}_beamsize{Params.beam_size}_minlen{Params.min_tgt_len}_lennorm{Params.len_norm_factor}_ngram{Params.block_ngram_repeat}.json'
	log_file_path = os.path.join(output_dir, log_file_name)

	with open(log_file_path, 'w', encoding='utf8') as f:
		json.dump(training_log, f, indent=4, ensure_ascii=False)


def evaluate(model, dataloader, output_dir, device):
	if Params.quick_test:
		decode_one_sample(model, dataloader, device)

	print('\n\n')
	logger.info("***** Running evaluation *****")
	model.eval()

	eval_log = {
		'arguments': vars(Params),
		'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
		'finish_time': '',
		'rouge_score': {},
		'predict_log': []
	}

	# Decode samples
	all_labels = []
	all_predicts = []

	with torch.no_grad():
		for index, batch in enumerate(tqdm(dataloader, desc='Evaluation step', position=0, leave=True, ascii=True)):
			step = index + 1

			# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
			batch_guids = batch[0]
			batch_src_ids = batch[1]
			batch_src_mask = batch[2]
			batch_tgt_ids = batch[3]
			batch_tgt_mask = batch[4]
			batch_labels = tokenizer.batch_decode(batch_tgt_ids, skip_special_tokens=True)

			batch_outputs = model.beam_decode(
				batch_guids=batch_guids,
				batch_src_seq=batch_src_ids,
				batch_src_mask=batch_src_mask,
				beam_size=Params.beam_size,
				n_best=Params.n_best)

			batch_predict_ids = []

			for i in range(len(batch_guids)):
				best_hyp = batch_outputs[i][0]
				batch_predict_ids.append(best_hyp[1])

			batch_output_summaries = tokenizer.batch_decode(batch_predict_ids, skip_special_tokens=True)

			# Collect hypotheses
			for i in range(len(batch_guids)):
				beam_hyps = {}
    
				for h in range(len(batch_outputs[i])):
					beam_hyps[f'h{h + 1}'] = tokenizer.decode(batch_outputs[i][h][1], skip_special_tokens=True),

				eval_log['predict_log'].append({
					'guid': batch_guids[i].item(),
					'target': batch_labels[i],
					'predict': beam_hyps
				})

			all_labels = all_labels + batch_labels
			all_predicts = all_predicts + batch_output_summaries

			# Print output of the first sample in the batch
			if step % Params.print_predict_every == 0:
				guid = batch_guids[0]
				target_tokens = batch_labels[0]
				predict_tokens = batch_output_summaries[0]

				logger.info(f'Step Num: {step:,}')
				logger.info(f'Sample {guid}')
				logger.info(f'Target: {target_tokens}')
				logger.info(f'Predict: {predict_tokens}')
    
				save_eval_log(output_dir, eval_log)

			if Params.quick_test and step == 5:
				break

	# Doing word desegmentation before caculating ROUGE scores
	all_desegmented_labels = [text.replace('_', ' ') for text in all_labels]
	all_desegmented_predicts = [text.replace('_', ' ') for text in all_predicts]

	rouge_score = calculate_rouge(all_desegmented_labels, all_desegmented_predicts)
	eval_log['rouge_score'] = {
		'rouge1': round(rouge_score['rouge1'], 4),
		'rouge2': round(rouge_score['rouge2'], 4),
		'rougeL': round(rouge_score['rougeL'], 4)
	}

	eval_log['finish_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	save_eval_log(output_dir, eval_log)

	# Print metricts
	rouge1 = eval_log['rouge_score']['rouge1']
	rouge2 = eval_log['rouge_score']['rouge2']
	rougeL = eval_log['rouge_score']['rougeL']

	logger.info('*** Evaluation results ***')
	logger.info(f'Rouge-1: {rouge1}')
	logger.info(f'Rouge-2: {rouge2}')
	logger.info(f'Rouge-L: {rougeL}')
	logger.info('Evaluation finished.')


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info(f'Using device: {device}')
	
	# Setup output path
	output_dir = os.path.dirname(Params.model_path)
	logger.info(f'Evaluation output dir: {output_dir}')

	model = load_model(device)
	dataloader = load_data()

	evaluate(model, dataloader=dataloader, output_dir=output_dir, device=device)
	logger.info('Finished.')
	sys.exit(0)