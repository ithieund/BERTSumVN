# BERTSumVN

Vietnamese Abstractive Summarization using pre-trained BERT models.

## Description

This code is for experiments from the paper which is accepted by ICIC-ELB in May 2023.

Title: EXPLORING ABSTRACTIVE SUMMARIZATION METHODS FOR VIETNAMESE USING PRE-TRAINED BERT MODELS
Authors: Duy Hieu Nguyen, Trong Nghia Hoang, Dien Dinh and Long H.B. Nguyen.

We release this code to motivate further research on Vietnamese Abstractive Summarization tasks.

If you use this code for your work, please cite our paper with the following info:

> Duy Hieu Nguyen, Trong Nghia Hoang, Dien Dinh and Long H.B. Nguyen. 2023. EXPLORING ABSTRACTIVE SUMMARIZATION METHODS FOR VIETNAMESE USING PRE-TRAINED BERT MODELS. In ICIC Express Letters Part B: Applications, Volume 14, Number 11, pages 1143-1151. DOI: 10.24507/icicelb.14.11.1143.

**Credits**:

* Some code are borrowed from the open source public implementation of using Transformer for Abstractive Summarization by brs1977 ([https://github.com/brs1977/BERT-Transformer-for-Summarization](https://github.com/brs1977/BERT-Transformer-for-Summarization))
* Raw dataset Wikilingual is downloaded at [https://github.com/esdurmus/Wikilingua](https://github.com/esdurmus/Wikilingua). The license is belong to the original authors.
* Raw dataset VietNews is downloaded at [https://github.com/ThanhChinhBK/vietnews](https://github.com/ThanhChinhBK/vietnews). The license is belong to the original authors.

## Datasets

The processed datasets for this work are named Wikilingua-Abs-Sum and VietNews-Abs-Sum, which are released at the following repos:

* Wikilingua-Abs-Sum: [https://huggingface.co/datasets/ithieund/viWikiHow-Abs-Sum](https://huggingface.co/datasets/ithieund/viWikiHow-Abs-Sum)
* VietNews-Abs-Sum: [https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum](https://huggingface.co/datasets/ithieund/VietNews-Abs-Sum)

All preprocessed data files are saved in both TSV and JSONL format. You can select your prefer format to handle the data easily for your further works.

If you use this preprocessed data for your work, please cite the original dataset's paper to give credits to the original author team. Beside that, please cite our paper too for our contribution to a clean and easy access dataset for Vietnamese Abstractive Summarization study.

## Data preprocessing

Here are some scripts that we used to conduct the data preprocessing:

### viWikiHow-Abs-Sum

* Convert RAW data to TSV and JSONL: [https://colab.research.google.com/drive/1f6l_cTlrUn3HiTBtQkhVox5NH_BdiyKm?usp=sharing](https://colab.research.google.com/drive/1f6l_cTlrUn3HiTBtQkhVox5NH_BdiyKm?usp=sharing)
* Data statistics: [https://colab.research.google.com/drive/1LK7_U06HiWOJf5lNiRXFwoGvfJLQDahL?usp=sharing](https://colab.research.google.com/drive/1LK7_U06HiWOJf5lNiRXFwoGvfJLQDahL?usp=sharing)
* Word-segmentation using RDRsegmenter: [https://colab.research.google.com/drive/1roGnE87x3CbdwPCU-kDYiMlqrsbG7xmq?usp=sharing](https://colab.research.google.com/drive/1roGnE87x3CbdwPCU-kDYiMlqrsbG7xmq?usp=sharing)
* Word-segmentation using UETsegmenter: [https://colab.research.google.com/drive/1MVWPH_vgi4lSd4BL95McWW1uK3O6XNTq?usp=sharing](https://colab.research.google.com/drive/1MVWPH_vgi4lSd4BL95McWW1uK3O6XNTq?usp=sharing)
* Word-segmentation using UITws: [https://colab.research.google.com/drive/1RMUwtUB7UwiobzW549FPDjc1OuWX7VdL?usp=sharing](https://colab.research.google.com/drive/1RMUwtUB7UwiobzW549FPDjc1OuWX7VdL?usp=sharing)

### VietNews-Abs-Sum

* Convert RAW data to TSV: [https://colab.research.google.com/drive/1GW6WbhZcBQ-IkhzqE9wrj8TrcgWlZcFY?usp=sharing](https://colab.research.google.com/drive/1GW6WbhZcBQ-IkhzqE9wrj8TrcgWlZcFY?usp=sharing)
* Desegmentation on the RAW dataset: [https://colab.research.google.com/drive/1LKN-NmQflQJe28J8CYHaqJ3EjX3DwU-A?usp=sharing](https://colab.research.google.com/drive/1LKN-NmQflQJe28J8CYHaqJ3EjX3DwU-A?usp=sharing)
* Data statistics on RAW dataset: [https://colab.research.google.com/drive/1VJExnwKCZF43G3nDy72gE_Csf356VBK5?usp=sharing](https://colab.research.google.com/drive/1VJExnwKCZF43G3nDy72gE_Csf356VBK5?usp=sharing)
* Filter duplicates: [https://colab.research.google.com/drive/1Q5Y4RA0o3xaFSpNPBRgjzpjbruckAkBG?usp=sharing](https://colab.research.google.com/drive/1Q5Y4RA0o3xaFSpNPBRgjzpjbruckAkBG?usp=sharing)
* Desegmentation on the filtered dataset: [https://colab.research.google.com/drive/1FyIqGbRHYSUBnltwWGR5GnnusjWVAMG_?usp=sharing](https://colab.research.google.com/drive/1FyIqGbRHYSUBnltwWGR5GnnusjWVAMG_?usp=sharing)
* Data statistics on filtered dataset: [https://colab.research.google.com/drive/1zP8nmRHNdmhqEXUJr0NkhUiJ7Ip3rrVq?usp=sharing](https://colab.research.google.com/drive/1zP8nmRHNdmhqEXUJr0NkhUiJ7Ip3rrVq?usp=sharing)
* Word-segmentation using RDRsegmenter: [https://colab.research.google.com/drive/1vfx5n85aHX-65bLFOSoAzBCIUh-m9905?usp=sharing](https://colab.research.google.com/drive/1vfx5n85aHX-65bLFOSoAzBCIUh-m9905?usp=sharing)
* Word-segmentation using UETsegmenter: [https://colab.research.google.com/drive/1IZ3GEVGIhFkVp7WoJADgqC-OWv__S2qy?usp=sharing](https://colab.research.google.com/drive/1IZ3GEVGIhFkVp7WoJADgqC-OWv__S2qy?usp=sharing)
* Word-segmentation using UITws: [https://colab.research.google.com/drive/1nstmPZLZZwvn2XnW9WGF_Nz3sNLFUUer?usp=sharing](https://colab.research.google.com/drive/1nstmPZLZZwvn2XnW9WGF_Nz3sNLFUUer?usp=sharing)
* Word-segmentation using RDRsegmenter on the RAW dataset: [https://colab.research.google.com/drive/1xAZKdwyrVPPG4aQQaKtmeBIMhVLnzj7A?usp=sharing](https://colab.research.google.com/drive/1xAZKdwyrVPPG4aQQaKtmeBIMhVLnzj7A?usp=sharing)
* Word-segmentation using UITws on the RAW dataset: [https://colab.research.google.com/drive/1kBfQkx7gWpaQ9UxINFSBeAeUWK1gz9FP?usp=sharing](https://colab.research.google.com/drive/1kBfQkx7gWpaQ9UxINFSBeAeUWK1gz9FP?usp=sharing)

## Setup server

### Requirements

We run our training and decoding scripts on an Ubuntu server which is installed with some libs and tools. Please follow these steps to make sure you can reproduce the paper results.

1. Install conda ([https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html](https://))
2. Create a new conda environment for this project: `conda create -n bertsumvn python=3.7`
3. Activate that environment for subsequence steps: `conda activate bertsumvn`
4. Install gdown inside the activated environment: `pip install gdown`
5. Make sure you have installed the correct CUDA TOOLKIT: `nvidia-smi`

### Setup working directory

To run the scripts, we must download this repo into the Unbuntu server and pull the processed data into the data sub-directory inside the working directory. Here are some tips:

1. Download the repo manual and extract the ZIP file, or clone it using the GIT command: `git clone https://github.com/ithieund/BERTSumVN.git`
2. Use cd command to move into the working directory: `cd BERTSumVN`
3. Install required python modules: `pip install -r requirements.txt`
4. Create sub-directory to contain data: `mkdir data`
5. Download the processed data from the repo links above and extract them into the `BERTSumVN/data` directory

## Train models

Script for training is located at `BERTSumVN/script_train_model.py`

Here are the supported parameters:


| Param name                  | Data type | Description                                                                                                                                                             |
| ----------------------------- | ----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| mode                        | string    | Set value to`train` for training                                                                                                                                        |
| visible_gpus                | string    | Specify which GPUs can be used to accelerate training, eg.`'0,1'`                                                                                                       |
| logger_debug                | string    | Whether the logger should print debug messages. Accepted value is`True` or `False`                                                                                      |
| quick_test                  | string    | Set value to`True` to run a quick test for quick debug                                                                                                                  |
| output_dir                  | string    | Specify the output directory to save training output. Default value is`output`                                                                                          |
| max_src_len                 | int       | Specify the max length of the article sequence (count by tokens). In our experiments, we set this value to`512`                                                         |
| max_tgt_len                 | int       | Specify the max length of the abstract sequence (count by tokens). In our experiments, we set this value to`60` for Wikilingua and `50` for VietNews dataset            |
| print_predict_every         | int       | Set the frequent to print the predict log while training                                                                                                                |
| bert_model                  | string    | Specify which BERT model to be employed as the Encoder                                                                                                                  |
| freeze_encoder              | string    | Left the default value to`True` to reproduce the results in this paper. We haven't conducted any trainable Encoder experiments!                                         |
| decoder_layers_num          | int       | Set the number of the transformer layers for the Decoder. Left default value to`8` to reproduce our results                                                             |
| decoder_attention_dim       | string    | Set the number of dimensions for the transformer layers of the Decoder. Left default value to`64` to reproduce our                                                      |
| train_batch_size            | int       | Specify the number of samples per batch for training. In our experiments, we set this value to`32`                                                                      |
| valid_batch_size            | int       | Specify the number of samples per batch for evaluation at the end of each epoch. In our experiments, we set this value to`32`                                           |
| num_train_epochs            | int       | Specify the number of epochs for training. In our experiments, we set this value to`100`                                                                                |
| learning_rate               | float     | Specify the learning rate for optimization. In our experiments, we set this value to`5e-5`                                                                              |
| num_warmup_steps            | int       | Specify the number of warmup step before applying learning rate schedule. In our experiments, we set this value to`0`                                                   |
| gradient_accumulation_steps | int       | Specify the number of steps to accumulate the gradient before back-propagation. In our experiments, we set this value to`2` for Wikilingua and`10` for VietNews dataset |
| resume_from_epoch           | int       | This is experimental parameter for resuming from the best checkpoint. We haven't tested yet                                                                             |
| resume_checkpoint_dir       | string    | This is experimental parameter for resuming from the best checkpoint. We haven't tested yet                                                                             |
| last_best_checkpoint        | int       | This is experimental parameter for resuming from the best checkpoint. We haven't tested yet                                                                             |
| last_best_eval_loss         | float     | This is experimental parameter for resuming from the best checkpoint. We haven't tested yet                                                                             |
| early_stopping_delta        | float     | Specify the delta hyper parameter for Early Stopping. In our experiments, we set this value to`0`                                                                       |
| early_stopping_patience     | int       | Specify the patience hyper parameter for Early Stopping. In our experiments, we set this value to`5`                                                                    |
| label_smoothing_factor      | float     | Specify the alpha hyper parameter for Label Smoothing. In our experiments, we set this value to`0.6`                                                                    |
| save_total_limit            | int       | Set the number of total checkpoints to be saved                                                                                                                         |
| log_loss_every_step         | int       | Set value to`True` to print the train loss in every step                                                                                                                |
| ddp_master_port             | string    | This is hyper parameter for Distributed Data Parallel technique to train on multiple GPUs. Just set any number for the PORT                                             |

**Caution**: In our experiments, we used only 1 GPUs per training process as we experienced slowing down when using multiple GPUs at once. This seems to be a bug of the Distributed Data Parallel technique 😕

## Evaluate models

Script for evaluation is located at `BERTSumVN/script_evaluate_model.py`

Here are the supported parameters:


| Param name          | Data type | Description                                                                                                                                                               |
| --------------------- | ----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| mode                | string    | Set value to`eval` for evaluation                                                                                                                                         |
| visible_gpus        | string    | Specify which GPUs can be used to accerlerate training. Eg.`'1'`                                                                                                          |
| logger_debug        | string    | Whether the logger should print debug messages. Accepted value is`True` or `False`                                                                                        |
| quick_test          | string    | Set value to`True` to run a quick test for quick debug                                                                                                                    |
| output_dir          | string    | Specify the output directory to save evaluation output. Default value is`output`                                                                                          |
| max_src_len         | int       | Specify the max length of the article sequence (count by tokens). In our experiments, we set this value to`512`                                                           |
| max_tgt_len         | int       | Specify the max length of the abstract sequence (count by tokens). In our experiments, we set this value to`60` for Wikilingua and `50` for VietNews dataset              |
| print_predict_every | int       | Set the frequent to print the predict log while predicting                                                                                                                |
| bert_model          | string    | Specify which BERT model to be employed as the Encoder. The model name must match with the same name that we used in the train process                                    |
| model_path          | string    | Specify the path to the saved model checkpoint to evaluate                                                                                                                |
| data_path           | string    | Specify the patch to the`test` set of the same dataset that we used to train the model                                                                                    |
| batch_size          | int       | Specify the number of samples per batch for evaluation. In our experiments, we set this value to`32`                                                                      |
| beam_size           | int       | Set the number of beams for Beam Search. In our experiments, we set this value to`3`                                                                                      |
| n_best              | int       | Set the number of best hypothesis to be returned by Beam Search. In our experiments, we set this value to`3`                                                              |
| min_tgt_len         | int       | Set the expected min output length of the generated summary. In our experiments, we set this value to`20`, `30`, and `40` as noted in the paper                           |
| len_norm_factor     | float     | Set the alpha hyper parameter for Length Normalization technique. In our experiments, we set this value to`0.6`                                                           |
| block_ngram_repeat  | int       | Set the number of repeated ngram to be blocked by N-Gram Blocking technique (proposed by Yang Liu) to prevent repeat tokens. In our experiments, we set this value to`2`. |

## Model name convention

* PhoBERT2TRANS: the Transformer model which has been replaced the Encoder with PhoBERT-base model (12 layers)
* PhoBERT-large2TRANS: the Transformer model which has been replaced the Encoder with PhoBERT-large model (24 layers)
* mBERT2TRANS: the Transformer model which has been replaced the Encoder with mBERT-base model (12 layers)

In our work, the Decoder layers are 8 for all experiments. The Window technique is to support PhoBERT-based abstractive models to handle 512 subword tokens. Therefore, it is only applied to PhoBERT2TRANS and PhoBERT-large2TRANS models, which introduces new names PhoBERT2TRANS + Window and PhoBERT-large2TRANS + Window.
Please refer to our paper for more details.

## Reproduce the paper results

We have stored all running commands, progress logs, and model checkpoints in this Drive storage: [https://drive.google.com/drive/folders/1-8wjk6pK2hD9Wf8WSBi4M_-ElFO8EvX-?usp=share_link](https://drive.google.com/drive/folders/1-8wjk6pK2hD9Wf8WSBi4M_-ElFO8EvX-?usp=share_link)

The training & evaluation commands can be found at: [https://drive.google.com/file/d/1MCDf5i7B3W6rkqWbcmnY0lkpbfsr-bbu/view?usp=share_link](https://drive.google.com/file/d/1MCDf5i7B3W6rkqWbcmnY0lkpbfsr-bbu/view?usp=share_link)

Our experiments were executed in an Ubuntu server with 4 A100 GPUs and 128GB of RAM, but we used only 1 GPU for each execution and sometimes the GPU was occupied by some other users at the same time. Therefore, the running time is poorly impact. It can be much more faster when you run on your own GPU.

* The average training time for each epoch is about 10-15 mins on Wikilingua and 40-50 mins on VietNews dataset.
* The average prediction time is about 2-3 hours on Wikilingua and 9-12 hours on VietNews dataset, depend on the min expected output length.

## Addition experiments for my Thesis

I have done some addition experiments to support my Thesis Defense on 27 Sep 2023. The experiments have been executed in two different servers:

* Server with 2 2080Ti GPUs (11GB each): I used this server to decode output summaries for Test set of Wikilingua dataset at minL50 with 2 models PhoBERT2TRANS + Window and mBERT2TRANS
* Server with 3 A100 GPUs (32GB each): I used this server to decode output summaries for Val set of Wikilingua and VietNews datasets at minL20 with 2 models PhoBERT2TRANS + Window and mBERT2TRANS. Besides, I have conducted training (on Train set) and decoding (on Val set at minL20) phases for model PhoBERT-large2TRANS with the 2 datasets Wikilingua and VietNews.

The additional model checkpoints and evaluation results can be found at: [https://drive.google.com/drive/folders/1C4XbIFt2pgsoyGttghXx8CMlL50dIi3K?usp=sharing](https://drive.google.com/drive/folders/1C4XbIFt2pgsoyGttghXx8CMlL50dIi3K?usp=sharing)

In these additional experiments, the GPUs are not shared to any other concurrent user. Therefore, the training time and decoding time statistics is more accurate than in the server with 4 A100 GPUs but with concurrent users above.

* The average training time for each epoch of the model PhoBERT-large2TRANS is about 4 mins on Wikilingua and 30 mins on VietNews dataset.
* The prediction time of the model PhoBERT2TRANS + Window is about 30 mins on Wikilingua and 4 hour 55 mins on VietNews dataset (at minL20).
* The prediction time of the models PhoBERT2TRANS + Window and mBERT2TRANS are about 1 hour 20 mins and 1 hour 50 mins on Wikilingua (at minL50).
