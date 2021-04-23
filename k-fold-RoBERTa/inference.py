from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device, batch_size=32):
    logits = []
    predictions = []

    dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
    model.eval()

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if 'token_type_ids' in data.keys():
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                )
            else:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                )

        _logits = outputs[0].detach().cpu().numpy()      
        _predictions = np.argmax(_logits, axis=-1)

        logits.append(_logits)
        predictions.extend(_predictions.ravel())

    return np.concatenate(logits), np.array(predictions)

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = 'xlm-roberta-large' 
  #tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
  tokenizer = XLMRobertaTokenizer.from_pretrained(TOK_NAME)

  # load my model
  MODEL_NAME = args.model_dir # model dir.
  model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  logits, predictions = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(predictions, columns=['pred'])
  output.to_csv('./prediction/submission.csv', index=False)
  np.save(os.path.join('./prediction/', r'logits.npy'), logits)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-6000")
  args = parser.parse_args()
  print(args)
  main(args)
  
