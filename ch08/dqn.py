import copy
from collections import deque
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# ============================================================
# 目的：CartPole に対して DQN（Deep Q-Network）を動かす最小例
# ============================================================
# このコードが実装しているアルゴリズムは DQN の基本形で、構成要素は次の通り。
#
# 1) Q関数の関数近似：
#    状態 $s$ を入力し、各行動の価値 $Q_\theta(s,a)$ を出力するニューラルネットを学習する。
#
# 2) TDターゲット（Q-learningのmaxバックアップ）：
#    1-step TDターゲットを
#      $y = r + \gamma (1-\text{done}) \max_{a'} Q_{\theta^-}(s',a')$
#    とする（終端では次項を0にする）。
#    ※ $\theta^-$ はターゲットネットワークのパラメータ。
#
# 3) 損失（回帰問題として学習）：
#    予測値 $Q_\theta(s,a)$ を TDターゲット $y$ に近づけるよう MSE を最小化する。
#
# 4) 経験再生（Replay Buffer）：
#    連続サンプルの相関を弱めるため、遷移を貯めてランダムにミニバッチ抽出して学習する。
#
# 5) ターゲットネットワーク（Target Network）：
#    ブートストラップ先（次状態価値）を一定期間固定し、学習を安定化させる。
#
# 注意（理論背景）：
# - 「関数近似 + ブートストラップ + オフポリシー」は不安定化し得る（deadly triad）。
#   DQNは Replay と Target Net で実用上かなり安定化している。
# ============================================================


# ------------------------------------------------------------
# ReplayBuffer：経験再生バッファ
# ------------------------------------------------------------
class ReplayBuffer:
    """
    経験再生（Experience Replay）のためのリングバッファ。

    保存する遷移（transition）は標準形：
      (s, a, r, s', done)

    ここで done は「この遷移でエピソードが終了したか」を表すフラグ。
    DQNのTDターゲットでは
      y = r + gamma * (1-done) * max_a' Q_target(s',a')
    のように (1-done) を掛けて終端でブートストラップを止めるため、
    done は 0/1 の数値として扱える形が便利。
    """

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        # state / next_state は観測ベクトル（np.ndarray想定）
        # action は int、reward は float、done は bool or 0/1
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        """
        一様ランダムにミニバッチをサンプルして numpy 配列で返す。

        実戦では prioritized replay などもあるが、最小例として一様サンプルにしている。
        """
        data = random.sample(self.buffer, self.batch_size)

        # 学習器に入れやすい形へ整形：
        # state, next_state: (B, obs_dim)
        # action, reward, done: (B,)
        state = np.stack([x[0] for x in data]).astype(np.float32)
        action = np.array([x[1] for x in data], dtype=np.int64)
        reward = np.array([x[2] for x in data], dtype=np.float32)
        next_state = np.stack([x[3] for x in data]).astype(np.float32)

        # done は 0/1 にしておくと (1-done) が書きやすい
        done = np.array([x[4] for x in data], dtype=np.float32)

        return state, action, reward, next_state, done


# ------------------------------------------------------------
# QNet：Q(s, a) を出すニューラルネット（Torch版）
# ------------------------------------------------------------
class QNet(nn.Module):
    """
    入力：状態ベクトル s（CartPoleは4次元）
    出力：各行動のQ値ベクトル [Q(s,0), Q(s,1), ...]
    """

    def __init__(self, state_dim: int, action_size: int):
        super().__init__()

        # 隠れ層 2層の MLP（DQNの最小例としてよく使う形）
        # CartPole程度ならこれで十分動くことが多い
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, state_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (B, action_size)
        return x


# ------------------------------------------------------------
# DQNAgent：DQNの学習・行動選択をまとめたクラス
# ------------------------------------------------------------
class DQNAgent:
    """
    DQNの中核ロジック：
    - ε-greedy で行動選択（探索と活用）
    - ReplayBuffer に遷移を貯める
    - ランダムミニバッチで TD誤差を回帰する
    - Target Net を一定間隔で同期する
    """

    def __init__(self, state_dim: int, action_size: int, device: torch.device):
        # ハイパーパラメータ（代表的な初期値）
        self.gamma = 0.98
        self.lr = 5e-4
        self.epsilon = 0.1

        self.buffer_size = 10_000
        self.batch_size = 32

        self.state_dim = state_dim
        self.action_size = action_size
        self.device = device

        # 経験再生
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        # オンラインネット（学習対象）とターゲットネット（固定ターゲット用）
        self.qnet = QNet(state_dim, action_size).to(self.device)
        self.qnet_target = QNet(state_dim, action_size).to(self.device)

        # 初期は同じ重みからスタート（同期）
        self.sync_qnet(hard=True)

        # 最適化器（AdamはDQNでよく使う）
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state: np.ndarray) -> int:
        """
        ε-greedy による行動選択。

        - 確率 ε：探索（ランダム）
        - 確率 1-ε：活用（Qが最大の行動）

        注意：
        DQNはオフポリシーなので、収集方策（ここではε-greedy）と、
        ターゲット（maxバックアップ）は一致していなくて良い（むしろそれがQ-learning）。
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        # NNで Q(s,·) を計算して argmax を取る（推論なので勾配不要）
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )  # (1, state_dim)
            qs = self.qnet(s)  # (1, action_size)
            action = int(torch.argmax(qs, dim=1).item())
        return action

    def update(self, state, action, reward, next_state, done):
        """
        遷移をバッファに追加し、十分溜まったら1回学習する。

        学習で最小化するのは TD誤差の二乗（MSE）：
          L = ( Q_theta(s,a) - y )^2

        TDターゲット y は Q-learning の max バックアップ：
          y = r + gamma * (1-done) * max_{a'} Q_target(s',a')
        """
        # まずは経験を貯める（学習より先）
        self.replay_buffer.add(state, action, reward, next_state, done)

        # バッファが少ないうちはミニバッチが作れないので学習しない
        if len(self.replay_buffer) < self.batch_size:
            return None

        # ミニバッチをサンプル
        state_b, action_b, reward_b, next_state_b, done_b = (
            self.replay_buffer.get_batch()
        )

        # numpy -> torch へ
        s = torch.tensor(
            state_b, dtype=torch.float32, device=self.device
        )  # (B, state_dim)
        a = torch.tensor(action_b, dtype=torch.int64, device=self.device)  # (B,)
        r = torch.tensor(reward_b, dtype=torch.float32, device=self.device)  # (B,)
        s2 = torch.tensor(
            next_state_b, dtype=torch.float32, device=self.device
        )  # (B, state_dim)
        d = torch.tensor(done_b, dtype=torch.float32, device=self.device)  # (B,)

        # 現在の Q(s,a) を取り出す
        # qnet(s): (B, action_size)
        # gatherで各サンプルの選択行動 a に対応する列を抜く：
        #   Q(s,a) = q_values[range(B), a]
        q_values = self.qnet(s)
        q = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # ターゲット側：max_{a'} Q_target(s',a') を計算（ターゲットなので勾配は流さない）
        with torch.no_grad():
            next_q_values = self.qnet_target(s2)  # (B, action_size)
            next_q_max = torch.max(next_q_values, dim=1).values  # (B,)

            # 終端ならブートストラップを止めるため (1-d) を掛ける
            target = r + (1.0 - d) * self.gamma * next_q_max  # (B,)

        # 損失（MSE）：回帰として Q(s,a) を target に近づける
        loss = F.mse_loss(q, target)

        # 勾配降下
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 学習モニタ用に損失を返す（任意）
        return float(loss.item())

    def sync_qnet(self, hard: bool = True, tau: float = 1.0):
        """
        ターゲットネットワークを同期する。

        - hard=True のとき：完全コピー（DQNの標準的実装）
          theta^- <- theta
        - hard=False のとき：ソフト更新（Polyak平均）
          theta^- <- (1-tau)*theta^- + tau*theta
          ただしこのコードでは基本 hard を使う想定。

        ※ 本スクリプトでは「一定エピソードごとに hard sync」する。
        """
        if hard:
            self.qnet_target.load_state_dict(self.qnet.state_dict())
            return

        # ソフト更新（必要なら使う）
        with torch.no_grad():
            for p_t, p in zip(self.qnet_target.parameters(), self.qnet.parameters()):
                p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


# ============================================================
# 学習ループ（CartPole）
# ============================================================

# 旧Gymではなく Gymnasium を使う（NumPy 2.0問題やAPI差分を回避）。
# さらに CartPole-v0 は古いので CartPole-v1 を使うのが推奨。
env = gym.make("CartPole-v1")

# CartPoleの観測次元と行動数を環境から取得（環境依存を減らす）
state_dim = int(env.observation_space.shape[0])  # 4
action_size = int(env.action_space.n)  # 2

# 実行デバイス（GPUがあればGPU、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(state_dim=state_dim, action_size=action_size, device=device)

episodes = 300
sync_interval = 20  # ターゲット同期の頻度（エピソード単位の簡易実装）

reward_history = []
loss_history = []

for episode in range(episodes):
    # Gymnasiumの reset() は (obs, info) を返す
    state, info = env.reset(seed=episode)

    terminated = False
    truncated = False

    total_reward = 0.0
    total_loss = 0.0
    loss_count = 0

    while not (terminated or truncated):
        # 行動選択（ε-greedy）
        action = agent.get_action(state)

        # Gymnasiumの step() は 5つ返す：
        # (next_state, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        # done は旧Gym互換の統合フラグ
        done = bool(terminated or truncated)

        # 1ステップ学習（バッファに追加→ミニバッチが作れたら更新）
        loss = agent.update(state, action, reward, next_state, done)
        if loss is not None:
            total_loss += loss
            loss_count += 1

        state = next_state
        total_reward += float(reward)

    # ターゲットネット同期（hard update）
    if episode % sync_interval == 0:
        agent.sync_qnet(hard=True)

    reward_history.append(total_reward)
    loss_history.append(total_loss / max(loss_count, 1))

    if episode % 10 == 0:
        print(
            f"episode: {episode}, total_reward: {total_reward:.1f}, avg_loss: {loss_history[-1]:.6f}, epsilon: {agent.epsilon:.3f}"
        )

env.close()

# ============================================================
# 学習曲線の可視化
# ============================================================
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.plot(range(len(reward_history)), reward_history)
plt.show()

plt.figure()
plt.xlabel("Episode")
plt.ylabel("Average TD Loss")
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# ============================================================
# 学習済み方策でプレイ（描画）
# ============================================================
# render_mode="human" を make() 時に指定するのが Gymnasium 流儀。
play_env = gym.make("CartPole-v1", render_mode="human")

# 評価時は探索を切って greedy にする
agent.epsilon = 0.0

state, info = play_env.reset(seed=0)
terminated = False
truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action = agent.get_action(state)
    state, reward, terminated, truncated, info = play_env.step(action)
    total_reward += float(reward)

play_env.close()
print("Total Reward (greedy):", total_reward)
