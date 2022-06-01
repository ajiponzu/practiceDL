from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

"""モデルの定義"""
class MLP(nn.Module):

  """コンストラクタ"""
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
    super().__init__()
    self.fcl = nn.Linear(input_dim, hidden_dim) # メンバ1, 隠れ層, シグモイドで活性化
    self.fcl2 = nn.Linear(hidden_dim, output_dim) # メンバ2, 出力層, シグモイドで活性化

  """順伝播関数"""
  def forward(self, x: np.ndarray) -> int:
    x = self.fcl(x)
    x = torch.sigmoid(x) # 活性化
    x = self.fcl2(x)
    x = torch.sigmoid(x) # 活性化
    return x

  """勾配降下アルゴリズムと誤差逆伝播によるパラメーターの更新処理"""
  def train_step(self, x: np.ndarray, t: np.ndarray, criterion: torch.nn.BCELoss, optimizer: torch.optim.SGD) -> any:

    self.train() # モデルを訓練(学習)モードにする
    outputs = self(x) # モデルの出力を取得
    loss = criterion(outputs, t) # 出力と正解ラベルの誤差から損失を取得
    optimizer.zero_grad() # 勾配を0で初期化（累積してしまうため）
    loss.backward()  # 逆伝播の処理(自動微分による勾配計算)
    optimizer.step() # 勾配降下法の更新式を適用してバイアス、重みを更新

    return loss

"""end"""
