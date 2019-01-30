import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('DataSet Titanic')
print('https://www.kaggle.com/c/titanic')
print('--------------------------------')
print('            Titanic             ')
print('     Learning from Disaster     ')
print('================================')
print(),print()

# Carregar dataset para a variável DataFrame
data = pd.read_csv('../input/train.csv', header=0)

# Criação de dicionário
cols = {}

# Definição de propriedades de um dicionário para os valores
# da coluna e o resultado de sobrevivência
# for col in list(data.columns.values):
#      cols[col] = data[[col, 'Survived']].groupby([col], as_index=False).mean()
#      str_colunas += col + ' ; '

# Análise de informação faltando na tabela
qtd_valores = np.product(data.shape)
qtd_vazios = data.isnull().sum().sum()
pct_vazios = qtd_vazios/qtd_valores*100

print('Porcentagem(%) de campos vazios: ' + str(pct_vazios)[:5])


data[['Age', 'Survived']].groupby(['Age'], as_index=True).sum().plot(kind="bar")
plt.figure(1)
plt.title('Sobreviventes por idade')
plt.xlabel('idade')
plt.ylabel('chance (%)')
plt.show()