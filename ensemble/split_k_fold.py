import pickle as pickle
import pandas as pd

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

def save_fold_n(n, num_of_folds, dataframe):
    idx = n - 1
    num_of_data = len(dataframe)
    fold_size = num_of_data // num_of_folds
    if num_of_data <= fold_size * idx:
        return
    
    fold_n_start_idx = fold_size * idx
    fold_n_end_idx = fold_size * (idx + 1)

    fold_n_test_data = dataframe.iloc[fold_n_start_idx:fold_n_end_idx].copy()
    fold_n_train_data = dataframe.iloc[:fold_n_start_idx].copy().append(dataframe.iloc[fold_n_end_idx:].copy())

    train_data_file_name = f"fold{n}train.tsv"
    test_data_file_name = f"fold{n}test.tsv"

    fold_n_test_data.to_csv(f'/opt/ml/k-fold-data/{test_data_file_name}', index=False, sep="\t")
    fold_n_train_data.to_csv(f'/opt/ml/k-fold-data/{train_data_file_name}', index=False, sep="\t")

if __name__ == "__main__":
    train_data = load_data("/opt/ml/input/data/train/train.tsv")
    shuffled_train_data = train_data.sample(frac=1).reset_index(drop=True)
    print(len(shuffled_train_data))

    num_of_folds = 6
    for i in range(1, num_of_folds + 1):
        save_fold_n(i, num_of_folds, shuffled_train_data)

    #load_fold(3)





def load_fold(n):
    train_data_file_name = f"fold{n}train.tsv"
    test_data_file_name = f"fold{n}test.tsv"

    train_dataset = pd.read_csv(f'/opt/ml/k-fold-data/{train_data_file_name}', delimiter='\t')
    test_dataset = pd.read_csv(f'/opt/ml/k-fold-data/{test_data_file_name}', delimiter='\t')
  
    return train_dataset, test_dataset
