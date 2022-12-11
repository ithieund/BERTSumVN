import sys
import argparse
from collections import namedtuple
from distutils.util import strtobool

def convert_dict_to_object(dict):
	object = namedtuple('Object', dict.keys())(*dict.values())
	return object

def _parse_arguments():
	executed_script_name = sys.argv[0]
	parser = argparse.ArgumentParser()
	
	if 'train_model' in executed_script_name or 'evaluate_model' in executed_script_name:
		parser.add_argument('-mode', type=str, default='train', choices=['train', 'eval'], help='Are we doing training or evaluating?')
		parser.add_argument('-visible_gpus', type=str, default='-1', help='Which GPUs can be used for training')
		parser.add_argument('-logger_debug', type=str, default='False', choices=['True', 'False'], help='Whether the logger should print debug messages')
		parser.add_argument('-quick_test', type=str, default='False', choices=['True', 'False'])
		parser.add_argument('-output_dir', type=str, default='outputs')
		parser.add_argument('-max_src_len', type=int, default=512)
		parser.add_argument('-max_tgt_len', type=int, default=50)
		parser.add_argument('-print_predict_every', type=int, default=20)

		if 'train_model' in executed_script_name:
			parser.add_argument('-bert_model', type=str, choices=['vinai/phobert-base', 'vinai/phobert-large', 'bert-base-multilingual-cased'], required=True, help='Which pretrained BERT model to use for finetuning')
			parser.add_argument('-freeze_encoder', type=str, default='True', choices=['True', 'False'], help='Whether the encoder should be frozen. If the value is False then the total trainable parameters are very large!')
			parser.add_argument('-decoder_layers_num', type=int, default=8, help='Vanilla decoder hyper parameter')
			parser.add_argument('-decoder_attention_dim', type=int, default=64, help='Vanilla decoder hyper parameter')		
			parser.add_argument('-train_data_path', type=str, required=True, help='Path to train data file. Ex: data/VietNews-Abs-Sum/train_desegmented_512tokens.pt')
			parser.add_argument('-valid_data_path', type=str, required=True, help='Path to train data file. Ex: data/VietNews-Abs-Sum/valid_desegmented_512tokens.pt')
			parser.add_argument('-train_batch_size', type=int, default=16, help='Set larger batch size if we have large GPU memory')
			parser.add_argument('-valid_batch_size', type=int, default=16, help='Set larger batch size if we have large GPU memory')
			parser.add_argument('-num_train_epochs', type=int, default=50, help='Max number of training epochs')
			parser.add_argument('-learning_rate', type=float, default=0.0001, help='Peak learning rate after warmup. Set 0 to disable warmup')
			parser.add_argument('-num_warmup_steps', type=int, default=1000, help='Num optimization steps to do warmup before applying learning rate schedule.')
			parser.add_argument('-gradient_accumulation_steps', type=int, default=8, help='Instead of calculating gradient and backpropagating it at every training step, we will wait until it reach the accummulation step to do that kind of calculation at once, which will speed up training a bit. But it will change the definition of the optimization step: num optimization steps = num steps / num gradient acc steps')
			parser.add_argument('-resume_from_epoch', type=int, default=0, help='Provide last epoch number to resume. Otherwise, leave default value 0')
			parser.add_argument('-resume_checkpoint_dir', type=str, default=None, help='Provide last training checkpoint dir to resume. Otherwise, leave default value None')
			parser.add_argument('-last_best_checkpoint', type=int, default=None, help='Provide last best checkpoint epoch number when resume training. Otherwise, leave default value None')
			parser.add_argument('-last_best_eval_score', type=float, default=None, help='Provide last best score when resume training from last epoch. Otherwise, leave default value None')
			parser.add_argument('-early_stopping_delta', type=float, default=0.0001)
			parser.add_argument('-early_stopping_patience', type=int, default=3, help='How many epochs to wait for early stopping when the current epoch has worst performance. Set 0 to disable early stopping')
			parser.add_argument('-label_smoothing_factor', type=float, default=0.1, help='Epsilon value for label smoothing. Set 0 to disable label smoothing')
			parser.add_argument('-save_total_limit', type=int, default=2, help='How many checkpoints to save at max')
			parser.add_argument('-log_loss_every_step', type=str, default='False', choices=['True', 'False'], help='Whether it should log train loss and learning rate every step. If the value is true then the log file may be larger than usual!')
			parser.add_argument('-ddp_master_port', type=str, default='9999', help='Which port for the master process of distributed data parallel service to listen on. Need to change this port for each seperate run on the same server to prevent port conflict!')

		if 'evaluate_model' in executed_script_name:
			parser.add_argument('-bert_model', type=str, required=True, choices=['vinai/phobert-base', 'vinai/phobert-large', 'bert-base-multilingual-cased'], help='Which pretrained BERT model that was used to train the model?')
			parser.add_argument('-model_path', type=str, required=True)
			parser.add_argument('-data_path', type=str, required=True,)
			parser.add_argument('-batch_size', type=int, default=1, help='Currently function beam_decode only support a batch with 1 sample. So batch size should always be 1')
			parser.add_argument('-beam_size', type=int, default=5)
			parser.add_argument('-n_best', type=int, default=3)
			parser.add_argument('-min_tgt_len', type=int, default=10, help='Define min output sequence length to prevent too short sequence. Set 0 to disable this rule')
			parser.add_argument('-len_norm_factor', type=float, default=0, help='Weight for Length normalization by Wu et al. Best value between 0.6 and 0.7 according to the paper. Set 0.0 to disable its effect')
			parser.add_argument('-cov_penalty_factor', type=float, default=0, help='Weight for Coverage penalty by Wu et al. Best value is 0.2 according to the paper. Set 0.0 to disable its effect')
			parser.add_argument('-block_ngram_repeat', type=int, default=0, help='How long for the repeated n-gram to be blocked? Set 0 to disable ngram blocking')
	elif executed_script_name == 'script_prepare_dataset.py':
		parser.add_argument('-visible_gpus', type=str, default='-1', help='Which GPUs can be used')
		parser.add_argument('-bert_model', type=str, required=True, choices=['vinai/phobert-base', 'bert-base-multilingual-cased'], help='Which tokenizer to use')
		parser.add_argument('-train_csv_path', type=str, required=True)
		parser.add_argument('-valid_csv_path', type=str, required=True)
		parser.add_argument('-test_csv_path', type=str, required=True)
		parser.add_argument('-max_rows', default=-1, type=int, help='How many rows to fetch at max')
		parser.add_argument('-max_src_len', type=int, required=True, default=512)
		parser.add_argument('-max_tgt_len', type=int, required=True, default=50)
		parser.add_argument('-logger_debug', type=str, default='False', choices=['True', 'False'], help='Whether the logger should print debug messages')
		parser.add_argument('-quick_test', type=str, default='False', choices=['True', 'False'])

	params = parser.parse_args()

	# Convert string to bool
	bool_attributes = ['logger_debug', 'quick_test', 'freeze_encoder', 'log_loss_every_step']	# Don't forget to update this list when the arguments change
 
	for attribute in bool_attributes:
		if hasattr(params, attribute):
			bool_value = bool(strtobool(getattr(params, attribute)))
			setattr(params, attribute, bool_value)

	# Just for quick testing when training
	if params.quick_test:
		if 'script_train_model' in executed_script_name:
			setattr(params, 'num_train_epochs', 5)
			setattr(params, 'train_batch_size', 2)
			setattr(params, 'valid_batch_size', 2)
			setattr(params, 'gradient_accumulation_steps', 2)
			setattr(params, 'save_total_limit', 2)
			setattr(params, 'early_stopping_delta', 0.001)
			setattr(params, 'early_stopping_patience', 2)
			setattr(params, 'print_predict_every', 1)
   
		if 'script_evaluate_model' in executed_script_name:
			setattr(params, 'batch_size', 2)
			setattr(params, 'beam_size', 2)
			setattr(params, 'n_best', 2)

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