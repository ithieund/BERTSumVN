import os
import sys
import random
import numpy as np
import torch
import json
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
from torch.nn.init import xavier_normal_
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformer_model import BertAbsSum
from transformer_preprocess import DataProcessor
from transformer_utils import *
from params_helper import Constants, Params
from tqdm import tqdm

# Setup logger
logger = get_logger(__name__)

BOS_TOKEN = Constants.BOS_TOKEN
EOS_TOKEN = Constants.EOS_TOKEN

# Check checkpoint directory
if Params.resume_from_epoch > 0 and (Params.resume_checkpoint_dir is None or Params.resume_checkpoint_dir == ''):
	print('Error: You should provide the checkpoint directory that contains checkpoints to resume!')
	sys.exit(0)

# Set visible GPUs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = Params.visible_gpus
num_gpus = torch.cuda.device_count()

# Init pretrained encoder model
tokenizer = AutoTokenizer.from_pretrained(Params.bert_model, local_files_only=False)
bert_model = AutoModel.from_pretrained(Params.bert_model, local_files_only=False)


def init_process(rank, world_size):
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '9999'

	backend = 'nccl' if torch.cuda.is_available() else 'gloo'

	dist.init_process_group(
		backend=backend,
		rank=rank,
		world_size=world_size
	)


def set_seed(seed):
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	else:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)


def get_model(rank, device, checkpoint, output_dir):
	logger.info(f'*** Getting model at rank {rank} ***')

	if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir is not None:
		# Load config
		config_path = os.path.join(Params.resume_checkpoint_dir, 'config.json')
		logger.info(f'Loading config from {config_path}')

		with open(config_path, 'r') as f:
			config = json.load(f)
	else:
		# Init config
		bert_config = bert_model.config
		decoder_config = {}
		decoder_config['vocab_size'] = bert_config.vocab_size
		decoder_config['d_word_vec'] = bert_config.hidden_size
		decoder_config['n_layers'] = Params.decoder_layers_num
		decoder_config['n_head'] = bert_config.num_attention_heads
		decoder_config['d_k'] = Params.decoder_attention_dim
		decoder_config['d_v'] = Params.decoder_attention_dim
		decoder_config['d_model'] = bert_config.hidden_size
		decoder_config['d_inner'] = decoder_config['d_model']

		config = {'bert_config': bert_config.__dict__, 'decoder_config': decoder_config}

	# Save config to output directory
	if (rank == 0):
		out_config_path = os.path.join(output_dir, 'config.json')
		logger.info(f'Saving config into {out_config_path}')

		with open(out_config_path, 'w') as f:
			json.dump(config, f, indent=4)

	config['freeze_encoder'] = Params.freeze_encoder
	model = BertAbsSum(bert_model_path=Params.bert_model, config=config, device=device)

	if checkpoint is not None:
		logger.info(f'Loading model from checkpoint')
		model.load_state_dict(checkpoint['model_state_dict'])

	model.to(device)

	if torch.cuda.is_available():
		if num_gpus > 1:
			model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
	else:
		model = DistributedDataParallel(model, find_unused_parameters=True)

	return model


def get_train_dataloader(rank, world_size):
	print('\n\n')
 
	logger.info(f'Loading prepared train data at: {Params.train_data_path}')
	processor = DataProcessor()

	train_dataset = torch.load(Params.train_data_path)

	if num_gpus == 1:
		train_dataloader = processor.create_dataloader(train_dataset, Params.train_batch_size)
	else:
		train_dataloader = processor.create_distributed_dataloader(rank, world_size, train_dataset, Params.train_batch_size)

	check_data(train_dataloader)
	return train_dataloader


def get_valid_dataloader(rank, world_size):
	print('\n\n')
	
	logger.info(f'Loading prepared valid data at: {Params.valid_data_path}')
	processor = DataProcessor()

	valid_dataset = torch.load(Params.valid_data_path)
	valid_dataloader = processor.create_dataloader(valid_dataset, Params.valid_batch_size)

	check_data(valid_dataloader)
	return valid_dataloader


def check_data(dataloader):
	print('\n\n')
	logger.info('*** Checking data ***')
	batch = next(iter(dataloader)) # Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
	batch_guids = batch[0]
	batch_src_ids = batch[1]
	batch_tgt_ids = batch[3]

	# for i in range(len(batch_guids)):
	for i in range(1):
		logger.info(f'Sample {batch_guids[i]}')
		logger.info(f'Source: {tokenizer.decode((batch_src_ids[i]), skip_special_tokens=True)}')
		logger.info(f'Target: {tokenizer.decode((batch_tgt_ids[i]), skip_special_tokens=True)}')

def cal_performance(logits, ground, smoothing=True):
	ground = ground[:, 1:]
	logits = logits.view(-1, logits.size(-1))
	ground = ground.contiguous().view(-1)

	loss = cal_loss(logits, ground, smoothing=smoothing)

	pad_mask = ground.ne(Constants.PAD)
	pred = logits.max(-1)[1]
	correct = pred.eq(ground)
	correct = correct.masked_select(pad_mask).sum().item()
	n_tokens = pad_mask.sum().item()
	return loss, correct, n_tokens


def cal_loss(logits, ground, smoothing=True):
	def label_smoothing(logits, labels):
		eps = 0.1
		num_classes = logits.size(-1)

		# >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
		# >>> z
		# tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
		#        [ 0.0000,  0.0000,  0.0000,  1.2300]])
		one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
		one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
		log_prb = F.log_softmax(logits, dim=1)
		non_pad_mask = ground.ne(Constants.PAD)
		loss = -(one_hot * log_prb).sum(dim=1)
		loss = loss.masked_select(non_pad_mask).mean()
		return loss

	if smoothing:
		loss = label_smoothing(logits, ground)
	else:
		loss = F.cross_entropy(logits, ground, ignore_index=Constants.PAD)
		# criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
		# loss = criterion(logits, ground)

	return loss


def init_parameters(model):
	for name, param in model.named_parameters():
		if 'encoder' not in name and 'tgt_embed' not in name and param.dim() > 1:
			xavier_normal_(param)


def do_validate(valid_dataloader, model, device, epoch_no):
	valid_iterator = tqdm(valid_dataloader, desc=f'Validate epoch {epoch_no}', position=0)
	model.eval()
	epoch_avg_valid_loss = 0
	total_valid_loss = 0
	valid_steps_num = 0

	with torch.no_grad():
		for step, batch in enumerate(valid_iterator):
			step = step + 1

			# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
			batch_guids = batch[0]
			batch_src_ids = batch[1].to(device)
			batch_src_mask = batch[2].to(device)
			batch_tgt_ids = batch[3].to(device)
			batch_tgt_mask = batch[4].to(device)

			logits = model.forward(
				batch_src_seq=batch_src_ids,
				batch_src_mask=batch_src_mask,
				batch_tgt_seq=batch_tgt_ids,
				batch_tgt_mask=batch_tgt_mask)
			loss, n_correct, n_tokens = cal_performance(logits, batch_tgt_ids)

			# del batch_src_ids
			# del batch_src_mask
			# del batch_tgt_mask
			# gc.collect()
			# torch.cuda.empty_cache()

			total_valid_loss += loss.item()
			valid_steps_num += 1

			valid_iterator.set_postfix({'Loss': loss.item(), 'Correct': f'{n_correct}/{n_tokens}'})

			# del batch_tgt_ids
			# del logits
			# del loss
			# gc.collect()
			# torch.cuda.empty_cache()

			if step == 5 and Params.quick_test:
				break   # For quick testing

		epoch_avg_valid_loss = total_valid_loss / valid_steps_num

	return epoch_avg_valid_loss


def save_training_log(output_dir, training_log):
	with open(os.path.join(output_dir, f'training_log.json'), 'w', encoding='utf8') as f:
		json.dump(training_log, f, indent=4, ensure_ascii=False)


def cleanup_older_checkpoints(output_dir, current_epoch):
	older_checkpoint_num = current_epoch - Params.save_total_limit

	if older_checkpoint_num < 1:
		return	# Nothing to do for now

	for i in range(1, older_checkpoint_num + 1):
		checkpoint_filepath = os.path.join(output_dir, f'Checkpoint_{i}.pt')
		
		if os.path.isfile(checkpoint_filepath):
			os.remove(checkpoint_filepath)
			logger.info(f'Deleted checkpoint file: {checkpoint_filepath}')


def train(rank, world_size, output_dir):
	print('\n\n')
	init_process(rank, world_size)
	set_seed(0)

	device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
	checkpoint = None

	logger.info(f'Rank {rank}/{world_size} training process initialized.')

	train_dataloader = get_train_dataloader(rank, world_size)
	valid_dataloader = get_valid_dataloader(rank, world_size)

	if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir is not None:
		checkpoint_path = os.path.join(Params.resume_checkpoint_dir, f'Checkpoint_{Params.resume_from_epoch}.pt')
		logger.info(f'Loading checkpoint {checkpoint_path}')
		checkpoint = torch.load(checkpoint_path, map_location=device)

		if checkpoint['epoch'] != Params.resume_from_epoch:
			print('Error: Checkpoint epoch number mismatched!')
		else:
			logger.info(f'Check point valid. Last loss = {checkpoint["loss"]}')

	if num_gpus != 1:
		dist.barrier()
		logger.info(f'Rank {rank}/{world_size} training process passed blocking barrier.')
	
	model = get_model(rank, device, checkpoint, output_dir)
	init_parameters(model)

	optimizer = AdamW(model.parameters(), lr=Params.learning_rate, correct_bias=False)
 
	if checkpoint is not None:
		logger.info(f'Loading optimizer from checkpoint')
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	num_train_samples = len(train_dataloader.dataset)
	num_train_steps_total = int(num_train_samples / Params.train_batch_size) * Params.num_train_epochs
	num_train_optimization_steps = int(num_train_steps_total / Params.gradient_accumulation_steps)

	num_valid_samples = len(valid_dataloader.dataset)
	num_valid_steps = int(num_valid_samples / Params.valid_batch_size)

	if Params.num_warmup_steps > 0:
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=Params.num_warmup_steps,
			num_training_steps=num_train_optimization_steps,
			last_epoch=Params.resume_from_epoch - 1
		)

	if rank == 0:
		print('\n\n')
		logger.info('***** Running training *****')
		logger.info(f'  Num train samples = {num_train_samples:,}')
		logger.info(f'  Num train batch size = {Params.train_batch_size}')
		logger.info(f'  Num train steps total = {num_train_steps_total:,}')
		logger.info(f'  Num train optimization steps = {num_train_optimization_steps:,}')
		logger.info(f'  Num train epochs = {Params.num_train_epochs}')
		logger.info(f'  Num valid samples = {num_valid_samples:,}')
		logger.info(f'  Num valid batch size = {Params.valid_batch_size}')
		logger.info(f'  Num valid steps = {num_valid_steps:,}')

		if Params.resume_from_epoch > 0:
			logger.info('*** Resume training from epoch %d ***', Params.resume_from_epoch)

	model.train()
	global_step = 0
	best_model_checkpoint = Params.last_best_checkpoint
	best_eval_score = Params.last_best_eval_score
	early_stop_counter = 0	# Early stopping will be activated when the counter >= patient

	training_log = {
		'arguments': vars(Params),
		'best_model': {
			'epoch_no': 0,
			'eval_score': 0
		},
  		'checkpoints': [],
		'predict_log': []
	}

	if Params.log_loss_every_step:
		training_log['loss_log'] = []

	# Load training log from resume checkpoint directory
	if Params.resume_from_epoch > 0 and Params.resume_checkpoint_dir is not None:
		with open(os.path.join(Params.resume_checkpoint_dir, f'training_log.json'), 'r') as f:
			training_log = json.load(f)

	for epoch_no in range(1, Params.num_train_epochs + 1):
		epoch_no = Params.resume_from_epoch + epoch_no

		# do training
		model.train()
		total_train_loss = 0
		train_examples_num, train_steps_num = 0, 0
		train_iterator = tqdm(train_dataloader, desc=f'Training epoch {epoch_no}', position=0)

		for step, batch in enumerate(train_iterator):
			step = step + 1
			global_step += 1

			# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
			batch_guids = batch[0]
			batch_src_ids = batch[1].to(device)
			batch_src_mask = batch[2].to(device)
			batch_tgt_ids = batch[3].to(device)
			batch_tgt_mask = batch[4].to(device)

			logits = model.forward(
				batch_src_seq=batch_src_ids,
				batch_src_mask=batch_src_mask,
				batch_tgt_seq=batch_tgt_ids,
				batch_tgt_mask=batch_tgt_mask)
			loss, n_correct, n_tokens = cal_performance(logits, batch_tgt_ids)

			# del batch_src_ids
			# del batch_src_mask
			# del batch_tgt_mask
			# gc.collect()
			# torch.cuda.empty_cache()

			actual_loss = loss.item()
			learning_rate = optimizer.param_groups[0]['lr']

			if Params.gradient_accumulation_steps > 1:
				loss = loss / Params.gradient_accumulation_steps

			loss.backward()
			# nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			if step % Params.gradient_accumulation_steps == 0:
				optimizer.step()
				
				if Params.num_warmup_steps > 0:
					scheduler.step()
				
				optimizer.zero_grad()

			if rank == 0:
				train_iterator.set_postfix({'Loss': actual_loss, 'Correct': f'{n_correct}/{n_tokens}'})

			if rank == 0 and step % Params.print_predict_every == 0:
				guid = batch_guids[0].item()
				target_tokens = tokenizer.decode(batch_tgt_ids[0].cpu().numpy(), skip_special_tokens=True)
				predict_tokens = tokenizer.decode(logits[0].max(-1)[1].cpu().numpy(), skip_special_tokens=True)

				logger.info(f'Epoch: {epoch_no}, Step: {step:,}, Loss: {actual_loss}, Learning Rate: {learning_rate}.')
				logger.info(f'Global steps: {global_step:,}')
				logger.info(f'Sample {guid}')
				logger.info(f'Target: {target_tokens}')
				logger.info(f'Predict: {predict_tokens}')

				training_log['predict_log'].append({
					'epoch_no': epoch_no,
					'global_step': global_step,
					'train_loss': actual_loss,
					'learning_rate': learning_rate,
					'predict': {
						'sample_guid': guid,
						'target_tokens': target_tokens,
						'predict_tokens': predict_tokens
					}
				})

				save_training_log(output_dir, training_log)
	
			if rank == 0:
				# Write training log every step
				if Params.log_loss_every_step:
					training_log['loss_log'].append({
						'epoch_no': epoch_no,
						'global_step': global_step,
						'train_loss': actual_loss,
						'learning_rate': learning_rate
					})

					save_training_log(output_dir, training_log)

			# del batch_tgt_ids
			# del logits
			# del loss
			# gc.collect()
			# torch.cuda.empty_cache()

			total_train_loss += actual_loss
			train_examples_num += len(batch_guids)
			train_steps_num += 1

			if step == 10 and Params.quick_test:
				break   # For quick testing

		epoch_avg_loss = total_train_loss / train_steps_num

		if rank == 0 and Params.output_dir is not None:
			logger.info(f'Saving checkpoint into {output_dir}')
			
			torch.save({
				'epoch': epoch_no,
				'model_state_dict': model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': epoch_avg_loss,
			}, os.path.join(output_dir, f'Checkpoint_{epoch_no}.pt'))

			logger.info('Checkpoint saved')

			# Remove older checkpoints to preserve only max save_total_limit checkpoints to save storage space
			if epoch_no > Params.save_total_limit:
				cleanup_older_checkpoints(output_dir=output_dir, current_epoch=epoch_no)

		# Do validation
		if rank == 0:
			val_loss = do_validate(
				valid_dataloader=valid_dataloader,
				model=model,
				device=device,
				epoch_no=epoch_no)

			training_log['checkpoints'].append({
				'epoch_no': epoch_no,
				'train_loss': actual_loss,
				'val_loss': val_loss,
				'finished_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			})

			logger.info(f'Epoch {epoch_no} valid loss: {val_loss}')
			logger.info(f'Epoch {epoch_no} finished.')

			# Find the best model in term of eval loss
			current_eval_score = -val_loss

			if best_eval_score is None:
				best_model_checkpoint = epoch_no
				best_eval_score = current_eval_score

				training_log['best_model']['epoch_no'] = best_model_checkpoint
				training_log['best_model']['eval_score'] = best_eval_score

				logger.info(f'Saving best model of epoch {best_model_checkpoint} with best score {-best_eval_score}')
				shutil.copyfile(os.path.join(output_dir, f'Checkpoint_{best_model_checkpoint}.pt'), os.path.join(output_dir, f'Best_Checkpoint.pt'))
			elif (current_eval_score - best_eval_score) < Params.early_stopping_delta:
				early_stop_counter += 1
			else:
				early_stop_counter = 0
				best_model_checkpoint = epoch_no
				best_eval_score = current_eval_score

				training_log['best_model']['epoch_no'] = best_model_checkpoint
				training_log['best_model']['eval_score'] = best_eval_score

				logger.info(f'Saving best model of epoch {best_model_checkpoint} with best score {-best_eval_score}')
				shutil.copyfile(os.path.join(output_dir, f'Checkpoint_{best_model_checkpoint}.pt'), os.path.join(output_dir, f'Best_Checkpoint.pt'))

			# Write training log
			save_training_log(output_dir, training_log)

			if early_stop_counter > 0:
				logger.info(f'Early stopping counter: {early_stop_counter}/{Params.early_stopping_patient}')

			# Activate early stopping
			if early_stop_counter >= Params.early_stopping_patient:
				logger.info(f'Early stopping at epoch {epoch_no}')
				# logger.info(f'Saving best model of epoch {best_model_checkpoint} with best score {-best_eval_score}')
				# shutil.copyfile(os.path.join(output_dir, f'Checkpoint_{epoch_no}.pt'), os.path.join(output_dir, f'Best_Checkpoint.pt'))
				break

	if rank == 0:
		logger.info('Training finished')
		sys.exit(0)


def cleanup_on_error(output_dir):
	if len(os.listdir(output_dir)) < 2:
		logger.info('Clearing output directory!')
		shutil.rmtree(output_dir)

	os.system('pkill -f multiprocessing.spawn')


if __name__ == '__main__':
	if torch.cuda.is_available():
		WORLD_SIZE = torch.cuda.device_count()
	else:
		WORLD_SIZE = 2	# Testing with CPU
	
	if torch.cuda.device_count() > 1:
		logger.info(f'Parallel training using {torch.cuda.device_count()} GPUs!')

	try:
		# Setup output path
		normalized_model_name = Params.bert_model.replace('/', '_')
		output_dir = os.path.join(Params.output_dir, datetime.now().strftime(f'model_{normalized_model_name}_{Params.max_src_len}_%Y.%m.%d_%H.%M.%S'))
		os.makedirs(output_dir, exist_ok=True)
		logger.info(f'Model output dir: {output_dir}')

		if num_gpus == 1:
			# Do single process training
			train(0, 1, output_dir=output_dir)
		else:
			# Do multiprocess training
			mp.spawn(
				train,
				args=(WORLD_SIZE, output_dir),
				nprocs=WORLD_SIZE,
				join=True
			)
	except KeyboardInterrupt:
		logger.info('Training interrupted by trainer!')
		cleanup_on_error(output_dir)
		sys.exit(0)
	except BaseException as error:
		logger.info('Program error!')
		cleanup_on_error(output_dir)
		raise error