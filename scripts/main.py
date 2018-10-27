import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('../input/train.csv', header=0)

# Chance de sobrevivência por classe
# train.groupby(['Pclass'], as_index=True)[['Survived']].mean().plot(kind='bar')

