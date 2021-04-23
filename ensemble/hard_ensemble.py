import os
import pandas as pd
from collections import Counter

# 이전에 생성해둔 submission용 csv 파일들을 가져온다
output1 = pd.read_csv(os.path.join('./prediction/', r'koelectra-submission.csv'))
output2 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission.csv'))
output3 = pd.read_csv(os.path.join('./prediction/', r'kobert-submission.csv'))
output4 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission2.csv'))
output5 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission3.csv'))
output6 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission4.csv'))
output7 = pd.read_csv(os.path.join('./prediction/', r'kobert-submission2.csv'))
output8 = pd.read_csv(os.path.join('./prediction/', r'kobert-submission3.csv'))
output9 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission5.csv'))
output10 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission6.csv'))
output11 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission7.csv'))
output12 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission8.csv'))
output13 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission9.csv'))
output14 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission10.csv'))
output15 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission11.csv'))
output16 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission12.csv'))
output17 = pd.read_csv(os.path.join('./prediction/', r'roberta-submission13.csv'))
output18 = pd.read_csv(os.path.join('./prediction/', r'koelectra-submission2.csv'))
output19 = pd.read_csv(os.path.join('./prediction/', r'koelectra-submission3.csv'))
output20 = pd.read_csv(os.path.join('./prediction/', r'koelectra-submission4.csv'))
output21 = pd.read_csv(os.path.join('./prediction/', r'koelectra-submission5.csv'))

all_ans = []

for i in range(len(output1)):
    outputs = [output1["pred"][i], output2["pred"][i], output3["pred"][i], output4["pred"][i], output5["pred"][i], output6["pred"][i], output7["pred"][i], output8["pred"][i], output9["pred"][i], output10["pred"][i], output11["pred"][i], output12["pred"][i], output13["pred"][i], output14["pred"][i], output15["pred"][i], output16["pred"][i], output17["pred"][i], output18["pred"][i], output19["pred"][i], output20["pred"][i], output21["pred"][i]]
    ensemble_ans = Counter(outputs).most_common(1)
    all_ans.append(ensemble_ans[0][0])

submission = pd.DataFrame(all_ans, columns=['pred'])
submission.to_csv(os.path.join('./prediction/', r'ensemble-submission.csv'), index=False)