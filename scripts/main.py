import pandas as pd

train = pd.read_csv('../input/train.csv', header=0)

print(train.info())