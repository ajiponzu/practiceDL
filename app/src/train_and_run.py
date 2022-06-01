from os import pread
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim

import my_model

"""訓練データと正解ラベルの用意"""
train = np.array([[0,0], [0,1], [1,0], [1,1]]) # 訓練データ
label = np.array([[0], [1], [1], [0]]) # 正解ラベル
train_x = torch.Tensor(train) # Pytorchではモデルに入力するためにtensorオブジェクトに変換する必要がある
train_y = torch.Tensor(label)
"""end"""

"""モデルの呼び出しと初期化"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使えるならcudaを使用
model = my_model.MLP(2, 2, 1).to(device)
# print(model)
print(device.type)
"""end"""

"""損失関数とオプティマイザーの生成"""
criterion = nn.BCELoss() # バイナリクロスエントロピー誤差の損失関数
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) # 勾配降下アルゴリズムを使用するオプティマイザーを生成
"""end"""

"""モデルを使用して学習する"""
epochs = 4000
for epoch in range(epochs):
  epoch_loss = 0.0 # 勾配のリセット

  train_x, train_y = train_x.to(device), train_y.to(device)

  loss = model.train_step(train_x, train_y, criterion, optimizer)
  epoch_loss += loss.item()

  if (epoch + 1) % 1000 == 0:
    print('epoch({}) loss: {:.4f}'.format(epoch+1, epoch_loss))
"""end"""

"""学習済みモデルで予測する"""
outputs = model(train_x)
print(outputs)
preds = (outputs.to('cpu').detach().numpy().copy() > 0.5).astype(np.int32)
print(preds)
"""end"""