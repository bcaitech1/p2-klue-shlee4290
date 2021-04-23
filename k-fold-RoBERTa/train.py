import pickle as pickle
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from load_data import *

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train():
  # load model and tokenizer
  #MODEL_NAME = "bert-base-multilingual-cased"
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  MODEL_NAME = 'xlm-roberta-large'
  tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
  print(tokenizer.tokenize("이순신은 조선 중기의 무신이다."))
  print(tokenizer.tokenize("아버지가방에들어가신다."))
  tokenized_str = tokenizer.tokenize("이순신은 조선 중기의 무신이다." + tokenizer.sep_token + "아버지가방에들어가신다.")
  print(tokenized_str)

  # load dataset
  dataset = load_data("/opt/ml/input/data/train/train.tsv")
  label = dataset['label'].values

  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42

  cv = StratifiedShuffleSplit(n_splits=5, test_size = 0.8, train_size= 0.2)
  for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):
    train_dataset = dataset.iloc[train_idx]
    val_dataset = dataset.iloc[val_idx]

    # tokenizing dataset
    train_dataset = tokenized_dataset(train_dataset, tokenizer)
    val_dataset = tokenized_dataset(val_dataset, tokenizer)

    train_y = label[train_idx]
    val_y = label[val_idx]

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_dataset, train_y)
    RE_valid_dataset = RE_Dataset(val_dataset, val_y)

    # setting model hyperparameter
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
    model.to(device)
    
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      save_total_limit=2,              # number of total save model.
      save_steps=400,                 # model saving step.
      num_train_epochs=10,              # total number of training epochs
      learning_rate=1e-5,               # learning_rate
      per_device_train_batch_size=16,  # batch size per device during training
      #per_device_eval_batch_size=8,   # batch size for evaluation
      warmup_steps=300,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps = 400,            # evaluation step.
      dataloader_num_workers=4,
      metric_for_best_model="accuracy",
      greater_is_better = True,
      label_smoothing_factor=0.5
    )
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_valid_dataset,             # evaluation dataset
      compute_metrics=compute_metrics,         # define metrics function
    )

    # train model
    trainer.train()

def set_seed(seed=4290):
  random.seed(seed)
  np.random.seed(seed)

  torch.manual_seed(seed)

  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def main():
  #set_seed()
  train()

if __name__ == '__main__':
  main()
