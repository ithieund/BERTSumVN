import torch.nn as nn
import torch
import operator
from torch.nn.functional import log_softmax
from transformer.Layers import DecoderLayer
from transformer.Models2 import get_non_pad_mask, get_sinusoid_encoding_table, get_attn_key_pad_mask, get_subsequent_mask
from transformers import AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformer_utils import *
from params_helper import Params, Constants

logger = get_logger(__name__)

class BertPositionEmbedding(nn.Module):
	def __init__(self, max_seq_len, hidden_size, padding_idx=0):
		super().__init__()

		sinusoid_encoding = get_sinusoid_encoding_table(
			n_position=max_seq_len + 1, # Add 1 for the first zero position
			d_hid=hidden_size,
			padding_idx=padding_idx)

		self.embedding = nn.Embedding.from_pretrained(embeddings=sinusoid_encoding, freeze=True)

	def forward(self, x):
		return self.embedding(x)


class BertDecoder(nn.Module):
	def __init__(self, config, device, dropout=0.1):
		super().__init__()

		self.device = device
		bert_config = BertConfig.from_dict(config['bert_config'])
		decoder_config = config['decoder_config']
		n_layers = decoder_config['n_layers']
		n_head = decoder_config['n_head']
		d_k = decoder_config['d_k']
		d_v = decoder_config['d_v']
		d_model = decoder_config['d_model']
		d_inner = decoder_config['d_inner']
		vocab_size = decoder_config['vocab_size']

		self.sequence_embedding = BertEmbeddings(config=bert_config)
		self.position_embedding = BertPositionEmbedding(max_seq_len=Constants.MAX_TGT_SEQ_LEN, hidden_size=d_model)
		self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
		self.last_linear = nn.Linear(in_features=d_model, out_features=vocab_size)

	@timing
	def forward(self, batch_src_seq, batch_enc_output, batch_tgt_seq):
		batch_tgt_seq = batch_tgt_seq.to(self.device)
		dec_slf_attn_list, dec_enc_attn_list = [], []

		# -- Prepare masks
		dec_non_pad_mask = get_non_pad_mask(batch_tgt_seq)

		slf_attn_mask_subseq = get_subsequent_mask(batch_tgt_seq)
		slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=batch_tgt_seq, seq_q=batch_tgt_seq)
		dec_slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

		dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=batch_src_seq, seq_q=batch_tgt_seq)

		batch_size, tgt_seq_len = batch_tgt_seq.size()
		batch_tgt_pos = torch.arange(1, tgt_seq_len + 1).unsqueeze(0).repeat(batch_size, 1).to(self.device)

		dec_output = self.sequence_embedding(batch_tgt_seq) + self.position_embedding(batch_tgt_pos)

		# -- Forward
		for dec_layer in self.layer_stack:
			dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
				dec_output, batch_enc_output,
				non_pad_mask=dec_non_pad_mask,
				slf_attn_mask=dec_slf_attn_mask,
				dec_enc_attn_mask=dec_enc_attn_mask)

			dec_slf_attn_list.append(dec_slf_attn)
			dec_enc_attn_list.append(dec_enc_attn)

		batch_logits = self.last_linear(dec_output)

		return batch_logits, dec_enc_attn_list


class BertAbsSum(nn.Module):
	def __init__(self, config, device):
		super().__init__()

		self.device = device
		self.config = config
		self.encoder = AutoModel.from_pretrained(config['bert_model'])

		# Freeze encoder
		if config['freeze_encoder'] == True:
			for param in self.encoder.parameters():
				param.requires_grad = False

		self.decoder = BertDecoder(config=config, device=device)

		# Count total params
		stats = self.get_model_stats()
		enc_params = stats['enc_params']
		dec_params = stats['dec_params']
		total_params = stats['total_params']
		logger.info(f'Encoder total parameters: {enc_params:,}')
		logger.info(f'Decoder total parameters: {dec_params:,}')
		logger.info(f'Total model parameters: {total_params:,}')
		
		if Params.mode == 'train':
			enc_trainable_params = stats['enc_trainable_params']
			dec_trainable_params = stats['dec_trainable_params']
			total_trainable_params = stats['total_trainable_params']
			logger.info(f'Encoder trainable parameters: {enc_trainable_params:,}')
			logger.info(f'Decoder trainable parameters: {dec_trainable_params:,}')
			logger.info(f'Total trainable parameters: {total_trainable_params:,}')

	def get_model_stats(self):
		enc_params = sum(p.numel() for p in self.encoder.parameters())
		dec_params = sum(p.numel() for p in self.decoder.parameters())
		total_params = enc_params + dec_params

		enc_trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
		dec_trainable_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
		total_trainable_params = enc_trainable_params + dec_trainable_params

		stats = {
			'enc_params': enc_params,
			'dec_params': dec_params,
			'total_params': total_params,
			'enc_trainable_params': enc_trainable_params,
			'dec_trainable_params': dec_trainable_params,
			'total_trainable_params': total_trainable_params,
		}

		return stats

	# @timing
	def forward(self, batch_src_seq, batch_src_mask, batch_tgt_seq, batch_tgt_mask):
		# src/tgt shape: (batch_size, seq_len)

		# shift right
		batch_tgt_seq = batch_tgt_seq[:, :-1]
		batch_tgt_mask = batch_tgt_mask[:, :-1]

		# TODO: pass invterval token_type_ids as addition info like PreSum
		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)  							# [batch_size, seq_len, hidden_size]	
		batch_logits, _ = self.decoder.forward(batch_src_seq=batch_src_seq, batch_enc_output=batch_enc_output, batch_tgt_seq=batch_tgt_seq)	# [batch_size, seq_len, vocab_size]
		return batch_logits

	def batch_encode_src_seq(self, batch_src_seq, batch_src_mask):
		# Use window to scan the full input sequence (max 256) if the model is PhoBERT and the input data is longer than 256 tokens
		if 'phobert' in Params.bert_model and Params.max_src_len > 256:
			window_size = 256
			batch_src_seq1 = batch_src_seq[:,:window_size]
			batch_src_seq2 = batch_src_seq[:,window_size:]
			batch_src_mask1 = batch_src_mask[:,:window_size]
			batch_src_mask2 = batch_src_mask[:,window_size:]
			
			batch_enc_output1 = self.encoder.forward(input_ids=batch_src_seq1, attention_mask=batch_src_mask1)[0]	# [batch_size, window_size, hidden_size]
			batch_enc_output2 = self.encoder.forward(input_ids=batch_src_seq2, attention_mask=batch_src_mask2)[0]	# [batch_size, window_size, hidden_size]
			batch_enc_output = torch.cat([batch_enc_output1, batch_enc_output2], dim=1)								# [batch_size, full_seq_len, hidden_size]
		# Otherwise, encode the input sequence as usual
		else:
			batch_enc_output = self.encoder.forward(input_ids=batch_src_seq, attention_mask=batch_src_mask)[0]  	# [batch_size, seq_len, hidden_size]

		return batch_enc_output

	@timing
	def greedy_decode(self, batch):
		# Batch is a tuple of tensors (guids, src_ids, scr_mask, tgt_ids, tgt_mask)
		batch_src_seq = batch[1].to(self.device)
		batch_src_mask = batch[2].to(self.device)

		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)	# [batch_size, seq_len, hidden_size]	
		batch_dec_seq = torch.full((batch_src_seq.size(0),), Constants.BOS, dtype=torch.long)
		batch_dec_seq = batch_dec_seq.unsqueeze(-1).type_as(batch_src_seq)

		for i in range(Constants.MAX_TGT_SEQ_LEN):
			output_logits, _ = self.decoder.forward(batch_src_seq=batch_src_seq, batch_enc_output=batch_enc_output, batch_tgt_seq=batch_dec_seq)
			dec_output = output_logits.max(-1)[1]
			dec_output = dec_output[:, -1]
			batch_dec_seq = torch.cat((batch_dec_seq, dec_output.unsqueeze(-1)), 1)

		return batch_dec_seq

	@timing
	def beam_decode(self, batch_guids, batch_src_seq, batch_src_mask, beam_size, n_best):
		batch_size = len(batch_guids)
		batch_src_seq = batch_src_seq.to(self.device)
		batch_src_mask = batch_src_mask.to(self.device)

		batch_enc_output = self.batch_encode_src_seq(batch_src_seq=batch_src_seq, batch_src_mask=batch_src_mask)	# [batch_size, seq_len, hidden_size]
		decoded_batch = []

		# Decoding goes through each sample in the batch
		for idx in range(batch_size):
			logger.debug(f'Decoding sample {batch_guids[idx]}')
			beam_src_seq = batch_src_seq[idx].unsqueeze(0).to(self.device)  # Batch with 1 sample
			beam_enc_output = batch_enc_output[idx].unsqueeze(0)   # Batch with 1 sample

			beams = []
			start_node = BeamSearchNode(prev_node=None, token_id=Constants.BOS, log_prob=0)
			beams.append((start_node.eval(), start_node))

			end_nodes = []

			# Start decoding process for each source sequence
			for step in range(Constants.MAX_TGT_SEQ_LEN):
				logger.debug(f'Decoding step {step} with {len(beams)} beams')
				candidates = []

				for score, node in beams:
					dec_seq = node.seq_tokens  # [id_1, id_2]
					beam_dec_seq = torch.LongTensor(dec_seq).unsqueeze(0)  # Batch with 1 sample
					beam_dec_seq.to(self.device)
				
					# Decode for one step using decoder
					logger.debug('Getting decoder logits')
					output_logits, output_attentions = self.decoder.forward(batch_src_seq=beam_src_seq, batch_enc_output=beam_enc_output, batch_tgt_seq=beam_dec_seq)
					log_probs = log_softmax(output_logits[:, -1][0], dim=-1)	# output_logits shape: (batch_size, seq_len, vocab_size); log_probs shape: (vocab_size)
					sorted_log_probs, sorted_indices = torch.sort(log_probs, dim=-1, descending=True)
					logger.debug('Logits sorted by log probs')

					# Collect top beam_size candidates for this beam instance
					candidate_count = 0
					i = 0

					while candidate_count < beam_size:
						logger.debug(f'Collecting candidate {candidate_count}')
						logger.debug(f'Hypothesis {i}')
						decoded_token = sorted_indices[i].item()
						log_prob = sorted_log_probs[i].item()
						i += 1

						next_node = BeamSearchNode(prev_node=node, token_id=decoded_token, log_prob=node.log_prob + log_prob)

						# Block ngram repeats
						if Params.block_ngram_repeat > 0:
							logger.debug('Checking repeat ngrams')
							ngrams = set()
							has_repeats = False
							gram = []

							for j in range(len(next_node.seq_tokens)):
								# A gram is combination of the last n tokens where n = block_ngram_repeat param
								gram = (gram + [next_node.seq_tokens[j]])[-Params.block_ngram_repeat:]
								
								# Skip the blocking if it is in the exclusion list
								# if set(gram) & self.exclusion_tokens:
								# 	continue
								
								# Repeats detected, we can break here
								if tuple(gram) in ngrams:
									logger.debug('Repeated ngram: ' + ' '.join(map(str, gram)))
									has_repeats = True
									break
								# No repeat for now, add this gram to the ngram set
								else:
									ngrams.add(tuple(gram))

							# Add penalty to this hypothesis to prevent it from expanding its path
							if has_repeats:
								penaltized_log_prob = next_node.log_prob + (-10e20)
								next_node.set_log_prob(penaltized_log_prob)
						
						# This candidate finished its path
						if decoded_token == Constants.EOS:
							logger.debug('End node found!')

							if Params.min_tgt_len > 0:
								# Collect this end node if it has valid sequence length
								if next_node.seq_len >= Params.min_tgt_len:
									end_nodes.append((next_node.eval(), next_node))
							else:
								end_nodes.append((next_node.eval(), next_node))
						# This candidate is still expanding
						else:
							candidates.append((next_node.eval(), next_node))
							candidate_count += 1

				# Stop decoding as we found enough end nodes
				if len(end_nodes) >= n_best:
					break

				# Collect beam_size best candidates for next time step decoding
				logger.debug(f'Candidates count: {len(candidates)}')
				sorted_candidates = sorted(candidates, key=operator.itemgetter(0), reverse=True)
				beams = []	# Reset beam list here to maintain only beam_size hypothesis
    
				for i in range(beam_size):
					beams.append(sorted_candidates[i])

			# Get n_best hypotheses at the end of decoding process
			logger.debug(f'Collecting {n_best} best hypotheses')
			best_hypotheses = []

			sorted_beams = sorted(beams, key=operator.itemgetter(0), reverse=True)

			if len(end_nodes) < n_best:
				for i in range(n_best - len(end_nodes)):
					end_nodes.append(sorted_beams[i])

			sorted_end_nodes = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)

			for i in range(n_best):
				score, end_node = sorted_end_nodes[i]
				best_hypotheses.append((score, end_node.seq_tokens))

			decoded_batch.append(best_hypotheses)

		return decoded_batch

class BeamSearchNode(object):
	def __init__(self, prev_node, token_id, log_prob):
		self.finished = False   # Determine if the hypothesis decoding is finished
		self.prev_node = prev_node
		self.token_id = token_id
		self.log_prob = log_prob

		if prev_node is None:
			self.seq_tokens = [token_id]
		else:
			self.seq_tokens = prev_node.seq_tokens + [token_id]

		self.seq_len = len(self.seq_tokens)

		if token_id == Constants.EOS:
			self.finished = True

	def set_log_prob(self, log_prob):
		self.log_prob = log_prob

	# Get beam sore with Wu's length normalization (https://arxiv.org/abs/1609.08144)
	# Adopt pytorch code from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/GNMT/seq2seq/inference/beam_search.py
	def eval(self):
		score = self.log_prob

		# Set Params.len_norm_factor = 0 will disable length normalization
		norm_const = 5
		length_norm = (norm_const + self.seq_len) / (norm_const + 1.0)
		length_norm = length_norm ** Params.len_norm_factor
		score = score / length_norm
  
		return score

	def __lt__(self, other):
		return self.seq_len < other.seq_len

	def __gt__(self, other):
		return self.seq_len > other.seq_len
