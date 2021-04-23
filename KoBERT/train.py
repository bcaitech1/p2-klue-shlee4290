import pickle as pickle
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertModel
from load_data import *
from tokenization_kobert import KoBertTokenizer
from split_k_fold import *

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
  MODEL_NAME = 'monologg/kobert'
  #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
  print(tokenizer.tokenize("이순신은 조선 중기의 무신이다."))
  print(tokenizer.tokenize("아버지가방에들어가신다."))

  # load dataset
  #train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  #dev_dataset = load_data("./dataset/train/dev.tsv")
  #train_label = train_dataset['label'].values
  #dev_label = dev_dataset['label'].values
  train_dataset, dev_dataset = load_fold(6)
  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  #train_dataset, dev_dataset = torch.utils.data.random_split(RE_train_dataset, [7000, 2000])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  bert_config = BertConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
  #model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=13,              # number of total save model.
    #load_best_model_at_end=True,
    save_steps=100,                 # model saving step.
    num_train_epochs=8,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=1000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 100,            # evaluation step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
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
  set_seed()
  train()

if __name__ == '__main__':
  main()


