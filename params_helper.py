import sys
import argparse
from collections import namedtuple

def convert_dict_to_object(dict):
	object = namedtuple('Object', dict.keys())(*dict.values())
	return object

def _parse_arguments():
	executed_script_name = sys.argv[0]
	parser = argparse.ArgumentParser()
	
	if 'train_model' in executed_script_name or 'evaluate_model' in executed_script_name:
		parser.add_argument('-mode', type=str, default='train', choices=['train', 'eval'], help='Are we doing training or evaluating?')
		parser.add_argument('-visible_gpus', type=str, default='-1', help='Which GPUs can be used for training')
		parser.add_argument('-logger_debug', type=bool, default=False, help='Whether the logger should print debug messages')
		parser.add_argument('-quick_test', type=bool, default=False)
		parser.add_argument('-output_dir', type=str, default='outputs')
		parser.add_argument('-max_src_len', type=int, default=512)
		parser.add_argument('-max_tgt_len', type=int, default=50)
		parser.add_argument('-freeze_encoder', type=bool, default=True, help='Whether the encoder should be frozen. If the value is False then the total trainable parameters are very large!')
		parser.add_argument('-decoder_layers_num', type=int, default=8, help='Vanilla decoder hyper parameter')
		parser.add_argument('-decoder_attention_dim', type=int, default=64, help='Vanilla decoder hyper parameter')
		parser.add_argument('-print_predict_every', type=int, default=100)
  
		if 'train_model' in executed_script_name:
			parser.add_argument('-bert_model', type=str, default='bert-base-multilingual-cased', help='Which pretrained BERT model to use for finetuning')
			parser.add_argument('-train_data_path', type=str, default='data/processed/vnds/train_no_segmentation.pt')
			parser.add_argument('-valid_data_path', type=str, default='data/processed/vnds/valid_no_segmentation.pt')
			parser.add_argument('-train_batch_size', type=int, default=16, help='Set larger batch size if we have large GPU memory')
			parser.add_argument('-valid_batch_size', type=int, default=16, help='Set larger batch size if we have large GPU memory')
			parser.add_argument('-num_train_epochs', type=int, default=200, help='Max number of training epochs')
			parser.add_argument('-learning_rate', type=float, default=0.0001, help='Peak learning rate after warmup')
			parser.add_argument('-num_warmup_steps', type=int, default=1000, help='Max warmup steps before applying learning rate schedule')
			parser.add_argument('-gradient_accumulation_steps', type=int, default=8)
			parser.add_argument('-resume_from_epoch', type=int, default=0, help='Provide last epoch number to resume. Otherwise, leave default value 0')
			parser.add_argument('-resume_checkpoint_dir', type=str, default=None, help='Provide last training checkpoint dir to resume. Otherwise, leave default value None')
			parser.add_argument('-last_best_checkpoint', type=int, default=None, help='Provide last best checkpoint epoch number when resume training. Otherwise, leave default value None')
			parser.add_argument('-last_best_eval_score', type=float, default=None, help='Provide last best score when resume training from last epoch. Otherwise, leave default value None')
			parser.add_argument('-early_stopping_delta', type=float, default=0.005)
			parser.add_argument('-early_stopping_patient', type=int, default=5, help='How many epoch to wait for early stopping when the training epoch has nothing optimized')
			parser.add_argument('-save_total_limit', type=int, default=5, help='How many checkpoint files to save at max')
			parser.add_argument('-log_loss_every_step', type=bool, default=False, help='Whether it should log train loss and learning rate every step. If the value is true then the log file may be larger than usual!')
  
		if 'evaluate_model' in executed_script_name:
			parser.add_argument('-bert_model', type=str, default='bert-base-multilingual-cased', help='Which pretrained BERT model that was used to train the model?')
			parser.add_argument('-model_path', type=str)			
			parser.add_argument('-config_path', type=str)
			parser.add_argument('-data_path', type=str)
			parser.add_argument('-batch_size', type=int, default=1)
			parser.add_argument('-beam_size', type=int, default=5)
			parser.add_argument('-n_best', type=int, default=3)
			parser.add_argument('-min_tgt_len', type=int, default=0, help='Define min output sequence length to prevent too short sequence')
			parser.add_argument('-block_ngram_repeat', type=int, default=0, help='How long for the repeated n-gram to be blocked?')
	elif executed_script_name == 'script_prepare_dataset.py':
		parser.add_argument('-visible_gpus', type=str, default='-1', help='Which GPUs can be used')
		parser.add_argument('-bert_model', required=True, type=str, default='bert-base-multilingual-cased', help='Which pretrained BERT model to use')
		parser.add_argument('-train_csv_path', required=True, type=str)
		parser.add_argument('-valid_csv_path', required=True, type=str)
		parser.add_argument('-test_csv_path', required=True, type=str)		
		parser.add_argument('-max_rows', default=-1, type=int, help='How many rows to fetch at max')
		parser.add_argument('-max_src_len', required=True, type=int, default=512)
		parser.add_argument('-max_tgt_len', required=True, type=int, default=50)
		parser.add_argument('-logger_debug', type=bool, default=False, help='Whether the logger should print debug messages')
		parser.add_argument('-quick_test', type=bool, default=False)

	params = parser.parse_args()

	# Just for quick testing when training
	if params.quick_test and executed_script_name == 'script_train_model.py':
		setattr(params, 'num_train_epochs', 5)
		setattr(params, 'train_batch_size', 2)
		setattr(params, 'valid_batch_size', 2)
		setattr(params, 'save_total_limit', 2)
		setattr(params, 'early_stopping_delta', 0.001)
		setattr(params, 'early_stopping_patient', 2)
		setattr(params, 'print_predict_every', 1)
  
	return params

def _get_constants(params):
	constants = {
		'PAD': None,
		'UNK': None,
		'BOS': None,
		'EOS': None,
		'MASK': None,
		'PAD_TOKEN': None,
		'UNK_TOKEN': None,
		'BOS_TOKEN': None,
		'EOS_TOKEN': None,
		'MAX_TGT_SEQ_LEN': params.max_tgt_len,
	}
	
	if 'phobert' in params.bert_model:
		constants['PAD'] = 1
		constants['UNK'] = 3
		constants['BOS'] = 0
		constants['EOS'] = 2
		constants['MASK'] = 64000
		constants['PAD_TOKEN'] = '<pad>'
		constants['UNK_TOKEN'] = '<unk>'
		constants['BOS_TOKEN'] = '<s>'
		constants['EOS_TOKEN'] = '</s>'
	elif 'bert-base-multilingual' in params.bert_model:
		constants['PAD'] = 0
		constants['UNK'] = 100
		constants['BOS'] = 101
		constants['EOS'] = 102
		constants['MASK'] = 103
		constants['PAD_TOKEN'] = '[PAD]'
		constants['UNK_TOKEN'] = '[UNK]'
		constants['BOS_TOKEN'] = '[CLS]'
		constants['EOS_TOKEN'] = '[SEP]'
  
	return convert_dict_to_object(constants)


Params = _parse_arguments()
print('Params: ', Params)

Constants = _get_constants(Params)
print('Constants: ', Constants)
print('\n')

 # Delete variable and functions after use to prevent calling from other place
del _parse_arguments
del _get_constants