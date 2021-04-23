import pandas as pd
import numpy as np
import os

output1 = np.load(os.path.join('./prediction/', r'koelectra-logits-1.npy'))
output2 = np.load(os.path.join('./prediction/', r'koelectra-logits-2.npy'))
output3= np.load(os.path.join('./prediction/', r'koelectra-logits-3.npy'))

print(output1)

soft_ensemble = output1 + output2 + output3
print(soft_ensemble)

result = []

for output in soft_ensemble:
    result.append(np.argmax(output))

submission = pd.DataFrame(result, columns=['pred'])
submission.to_csv('./prediction/koelectra-submission.csv', index=False)