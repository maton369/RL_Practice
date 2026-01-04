import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # 親ディレクトリを import 対象に追加（教材コードの都合）
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.gridworld import GridWorld


def one_hot(state):
    """
    GridWorld の状態 (y, x) を one-hot ベクトルに変換する。

    状態空間は 3x4 の 12 状態（壁セルもインデックスは持つが、遷移では弾かれる）として扱う。
    one-hot は「関数近似器（NN）に状態を入れるための特徴量」の最小例であり、
    本来は座標特徴量や畳み込み入力など、より構造を反映した表現も使える。

    返り値は shape=(1, 12) の numpy 配列（バッチ次元 1 を付与）。
    """
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)

    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0

    return vec[np.newaxis, :]  # (1, 12)


class QNet(nn.Module):
    """
    状態 -> 各行動の Q 値 を出力するニューラルネットワーク（DQNの最小形）。

    入力:  one-hot 状態ベクトル x（shape=(B, 12)）
    出力:  Q(s, a) を並べたベクトル（shape=(B, action_size) = (B, 4)）

    アルゴリズム的には「Q 関数の関数近似（approximation）」を行っており、
    表形式の Q-learning の Q(s,a) テーブルを NN に置き換えた形になる。

    注意：
    - この実装は経験再生（replay buffer）やターゲットネットワークを持たないため、
      一般的な DQN より不安定になりやすい（教材として最小にしている）。
    """

    def __init__(self, input_size=12, hidden_size=100, action_size=4):
        super().__init__()
        # 線形層1：状態特徴量 -> 隠れ表現
        self.l1 = nn.Linear(input_size, hidden_size)
        # 線形層2：隠れ表現 -> 各行動の Q 値
        self.l2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """
        順伝播：
        - ReLU により非線形性を導入し、状態ごとの価値構造を表現できるようにする。
        """
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    """
    NN で Q(s,a) を近似する Q-learning エージェント（最小構成）。

    表形式 Q-learning の更新は

        target = r + γ * max_{a'} Q(s', a')
        Q(s,a) <- Q(s,a) + α (target - Q(s,a))

    だが、関数近似では「Q を直接置き換える」ことはせず、
    典型的に「二乗誤差で target に近づくよう NN を学習する」形にする。

    ここでの損失（1サンプル）は

        L = (target - Q_theta(s,a))^2

    となり、勾配降下により θ を更新する。

    重要ポイント（semi-gradient / ブートストラップ）：
    - target は Q(s',a') を含むが、更新時は「定数」として扱うのが基本（detach / no_grad）。
      そうしないと同じネットで target 側にも勾配が流れて学習が不安定になりやすい。
    """

    def __init__(self, device=None):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        # 実行デバイス（GPUがあればcuda、なければcpu）
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Qネットワーク
        self.qnet = QNet(
            input_size=12, hidden_size=100, action_size=self.action_size
        ).to(self.device)

        # 最小例なので SGD を使用（Adam にすると収束しやすいことが多い）
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state_vec_np):
        """
        ε-greedy による行動選択。

        入力:
        - state_vec_np: one-hot 状態（numpy, shape=(1, 12)）

        出力:
        - action: {0,1,2,3} のいずれか

        注意：
        - greedy 側は Q(s,·) の argmax。
        - 学習初期は同値最大が多く、argmax が偏ることがある。
          教材目的ならOKだが、厳密にはランダムタイブレーク等も考えられる。
        """
        if np.random.rand() < self.epsilon:
            # 探索（ランダム行動）
            return np.random.choice(self.action_size)

        # 活用（NN の出力から argmax）
        with torch.no_grad():
            state = torch.from_numpy(state_vec_np).to(self.device)  # (1, 12)
            qs = self.qnet(state)  # (1, 4)
            action = int(torch.argmax(qs, dim=1).item())
        return action

    def update(self, state_np, action, reward, next_state_np, done):
        """
        1ステップ遷移 (s,a,r,s',done) で Q-learning の更新を行う（オンライン・逐次更新）。

        入力:
        - state_np: one-hot 状態 s（numpy, shape=(1, 12)）
        - action: 実行した行動 a（int）
        - reward: 得た報酬 r（float）
        - next_state_np: one-hot 次状態 s'（numpy, shape=(1, 12)）
        - done: 終端かどうか（bool）

        アルゴリズム的に重要な処理：
        - next 側の max Q(s',·) は target を作るために使うが、
          勾配は流さない（detach/no_grad）＝ semi-gradient TD。
        """
        # numpy -> torch
        state = torch.from_numpy(state_np).to(self.device)  # (1, 12)
        next_state = torch.from_numpy(next_state_np).to(self.device)  # (1, 12)
        reward_t = torch.tensor(
            float(reward), dtype=torch.float32, device=self.device
        )  # scalar

        # --- TDターゲットの構築（勾配を流さない） ---
        with torch.no_grad():
            if done:
                # 終端では将来価値は 0 とみなす
                next_q_max = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                # 次状態での最大 Q を使う（オフポリシーの max バックアップ）
                next_qs = self.qnet(next_state)  # (1, 4)
                next_q_max = torch.max(next_qs, dim=1).values[0]  # scalar

            # target = r + γ * max_a' Q(s',a')
            target = reward_t + self.gamma * next_q_max  # scalar

        # --- 現在の Q(s,a) を取り出す ---
        qs = self.qnet(state)  # (1, 4)
        q_sa = qs[0, action]  # scalar（選択した行動の Q）

        # --- 損失（MSE） ---
        # 1サンプルなので単純に二乗誤差でよい。一般にはミニバッチ化する。
        loss = F.mse_loss(q_sa, target)

        # --- 勾配降下で更新 ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 学習モニタ用にスカラーを返す
        return float(loss.item())


# ------------------------------------------------------------
# 実験：GridWorld で NN版 Q-learning を実行し、損失推移と学習後Qを可視化
# ------------------------------------------------------------
env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    # エピソード開始
    state = env.reset()
    state = one_hot(state)

    total_loss = 0.0
    cnt = 0
    done = False

    while not done:
        # ε-greedy で行動選択
        action = agent.get_action(state)

        # 環境遷移
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        # TD更新（オンライン）
        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1

        # 次へ
        state = next_state

    # エピソード内の平均損失（ステップ数で割る）
    average_loss = total_loss / max(cnt, 1)
    loss_history.append(average_loss)

# --- 損失の可視化 ---
plt.xlabel("episode")
plt.ylabel("loss")
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# ------------------------------------------------------------
# 学習後の Q(s,a) を辞書に落として可視化（env.render_q が dict を期待するため）
# ------------------------------------------------------------
Q = {}
agent.qnet.eval()  # 推論モード（このネットはBN/Dropout無いが、習慣として）
with torch.no_grad():
    for state in env.states():
        state_vec = torch.from_numpy(one_hot(state)).to(agent.device)  # (1, 12)
        qs = agent.qnet(state_vec)[0]  # (4,)
        for action in env.action_space:
            Q[state, action] = float(qs[action].item())

env.render_q(Q)
