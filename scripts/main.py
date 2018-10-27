import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def to_percentage(v):
    return v * 100

font = {
    'family': 'serif'
}

train = pd.read_csv('../input/train.csv', header=0)

# Chance de sobreviver por sexo
train.groupby(['Sex'], as_index=True)[['Survived']].mean().applymap(to_percentage).plot(kind='bar')
plt.figure(2)
plt.title('Chance de sobrevivência por sexo', fontdict=font)
plt.xlabel('sexo', fontdict=font)
plt.ylabel('chance (%)', fontdict=font)

# Chance de sobrevivência por classe
train.groupby(['Pclass'], as_index=True)[['Survived']].mean().applymap(to_percentage).plot(kind='bar')
plt.figure(1)
plt.title('Chance de sobrevivência por classe', fontdict=font)
plt.xlabel('classe', fontdict=font)
plt.ylabel('chance (%)', fontdict=font)

plt.show()
