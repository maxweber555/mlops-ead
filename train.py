
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from tensorflow import keras
import tensorflow
import random as python_random
import numpy as np
import random
import os
import tensorflow as tf


def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

# %% [markdown]
# # 2 - Fazendo a leitura do dataset e atribuindo às respectivas variáveis


# %%
data = pd.read_csv(
    'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')

# %% [markdown]
# # 3 - Preparando o dado antes de iniciar o treino do modelo

# %%
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

y_train = y_train - 1
y_test = y_test - 1

# %% [markdown]
# # 4 - Criando o modelo e adicionando as camadas

# %%
reset_seeds()
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1], )))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# %% [markdown]
# # 5 - Compilando o modelo
#

# %%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# %%

os.environ['MLFLOW_TRACKING_USERNAME'] = 'maxweber555'  # 'renansantosmendes'
# '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0304bf1fb7fc02b53812bd5a2c0b4807ef4788ec'
mlflow.set_tracking_uri(
    # 'https://dagshub.com/renansantosmendes/mlops-ead.mlflow')
    'https://dagshub.com/maxweber555/my-first-repo.mlflow')

mlflow.tensorflow.autolog(log_models=True,
                          log_input_examples=True,
                          log_model_signatures=True)

# %% [markdown]
# # 6 - Executando o treino do modelo

# %%
with mlflow.start_run(run_name='experiment_mlops_ead') as run:
    model.fit(X_train,
              y_train,
              epochs=50,
              validation_split=0.2,
              verbose=3)
