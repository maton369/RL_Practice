# ============================================================
# PyTorch + Gymnasium 版：REINFORCE（モンテカルロ方策勾配）の最小例
# ============================================================
# 目的：
#   CartPole（離散行動2択）に対して、確率的方策 πθ(a|s) を学習し、
#   エピソード報酬の期待値を最大化する（方策最適化）。
#
# アルゴリズム（REINFORCE）の核：
#   1) 方策 πθ でエピソード（軌跡）をサンプルする
#   2) 各時刻 t のリターン（割引収益） G_t を計算する
#   3) 勾配推定：
#        ∇θ J(θ) ≈ Σ_t  G_t ∇θ log πθ(a_t|s_t)
#      を用いて θ を更新する
#
# 損失として書くと（最大化を最小化に変換）：
#   L(θ) = - Σ_t  G_t log πθ(a_t|s_t)
#
# ポイント：
# - 「行動をサンプルする」ことで探索しつつ、良かった行動の確率を上げる方向に更新される。
# - 価値関数を使わない純粋なモンテカルロ法なので、分散が大きい（学習がブレやすい）。
#   実務では baseline（例：価値関数）を入れた Actor-Critic に発展させるのが定石。
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Gym は unmaintained なので Gymnasium に置き換える
# Gymnasium は API が少し違う：
# - reset() -> (obs, info)
# - step(a) -> (obs, reward, terminated, truncated, info)
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def plot_total_reward(reward_history: List[float]) -> None:
    """エピソードごとの合計報酬をプロットする（学習の進行確認用）。"""
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


class PolicyNet(nn.Module):
    """
    方策ネットワーク πθ(a|s) を表現するニューラルネット。

    入力：
      - 状態 s（CartPole の観測ベクトル：通常 4 次元）
    出力：
      - 行動の確率分布 πθ(·|s)（softmax で正規化された確率）

    実装の意図：
      - 出力を logits として返し、サンプリングは Categorical(logits=...) に任せる。
        （数値的に安定で、log_prob も簡単に取れる）
    """

    def __init__(self, state_dim: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),  # ここは logits（未正規化スコア）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (B, state_dim)
        # return: shape (B, action_size) の logits
        return self.net(x)


@dataclass
class Transition:
    """
    REINFORCE は（状態価値ではなく）エピソード全体のサンプルから更新するため、
    1ステップごとに最低限必要な情報を保存する。

    - log_prob：その時刻に実際に選んだ行動の log πθ(a_t|s_t)
               これが損失 -log_prob * G_t を通して勾配を運ぶ
    - reward  ：環境から得た報酬 r_{t+1}
    """

    log_prob: torch.Tensor
    reward: float


class ReinforceAgent:
    """
    REINFORCE（モンテカルロ方策勾配）エージェント。

    学習フロー：
      - 1エピソード分、(log_prob, reward) を蓄積
      - エピソード終了後、各時刻のリターン G_t を計算
      - 損失 L = - Σ_t G_t log_prob_t を作って逆伝播・更新

    実務上の小技（安定化）：
      - returns を正規化（平均0・分散1）すると学習が安定しやすいことが多い
        ※ 期待値の最大化という目的は理論的に保たれやすいが、厳密にはバイアスが入りうる点は理解しておく
    """

    def __init__(
        self,
        state_dim: int,
        action_size: int,
        gamma: float = 0.98,
        lr: float = 2e-4,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy = PolicyNet(state_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # エピソードの軌跡を保持するバッファ（オンポリシーなので毎エピソード消費）
        self.memory: List[Transition] = []

    def get_action(self, state: np.ndarray) -> int:
        """
        現在の方策 πθ(·|s) から行動をサンプルする。

        - Categorical(logits=...) は内部で softmax を取り、
          サンプルと log_prob を提供してくれる。
        - REINFORCE は「サンプルした行動」の log_prob が必要なので、ここで保存する。
        """
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        logits = self.policy(state_t)  # shape (1, action_size)

        dist = Categorical(logits=logits)  # 離散分布 πθ(·|s)
        action_t = dist.sample()  # 行動を確率的にサンプル
        log_prob_t = dist.log_prob(action_t)  # log πθ(a|s)

        # memory に保存して、エピソード終了後にまとめて更新する
        self.memory.append(Transition(log_prob=log_prob_t, reward=0.0))

        # Python int として返す（env.step に渡しやすい）
        return int(action_t.item())

    def add_reward_to_last(self, reward: float) -> None:
        """
        get_action() で 1ステップ分の Transition を先に追加しているため、
        env.step 後に得た reward を「直近の Transition」に埋める。
        """
        self.memory[-1].reward = float(reward)

    def update(self) -> float:
        """
        エピソード終了後に方策を1回更新する。

        G_t の計算：
          G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ...

        損失：
          L = - Σ_t G_t log πθ(a_t|s_t)

        戻り値：
          学習の目安として loss（float）を返す。
        """
        # -----------------------------
        # (1) リターン（割引収益）を後ろから計算
        # -----------------------------
        returns: List[float] = []
        G = 0.0
        for tr in reversed(self.memory):
            G = tr.reward + self.gamma * G
            returns.append(G)
        returns.reverse()  # 時刻順に戻す

        # Tensor 化（shape (T,)）
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # -----------------------------
        # (2) 分散を下げるために returns を正規化（任意だが効果が出やすい）
        # -----------------------------
        # 標準偏差がゼロに近い場合の除算を避けるため eps を足す
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # -----------------------------
        # (3) 損失を構成：- Σ_t (G_t * log_prob_t)
        # -----------------------------
        # 良い（G_t が大きい）行動の log_prob を増やす方向に更新される
        loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for tr, Gt in zip(self.memory, returns_t):
            loss = loss + (-tr.log_prob * Gt)

        # -----------------------------
        # (4) 逆伝播して更新
        # -----------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # オンポリシーなのでエピソードのデータは使い切り
        self.memory.clear()

        return float(loss.detach().cpu().item())


# ============================================================
# 学習ループ（Gymnasium API 対応）
# ============================================================
episodes = 3000

# CartPole は v1 が推奨（v0 は古い）
# render は学習中は基本不要（遅くなる）。評価時だけ render_mode="human" を使うのが定石。
env = gym.make("CartPole-v1")

# 観測・行動の次元を env から取得（ハードコードしない方が堅牢）
state_dim = int(np.prod(env.observation_space.shape))
action_size = int(env.action_space.n)

# device は必要なら "cuda" に変える（GPUがある場合）
agent = ReinforceAgent(
    state_dim=state_dim, action_size=action_size, gamma=0.98, lr=2e-4, device="cpu"
)

reward_history: List[float] = []
loss_history: List[float] = []

for episode in range(episodes):
    # Gymnasium: reset() -> (obs, info)
    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # 1) 方策から行動をサンプル
        action = agent.get_action(state)

        # 2) 環境を1ステップ進める
        # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        # 3) 直近の transition に報酬を保存
        agent.add_reward_to_last(reward)

        total_reward += float(reward)
        state = next_state

        # 終端条件（MDPの終端 or 時間制限など）
        done = bool(terminated or truncated)

    # エピソードが終わったら REINFORCE で方策を更新
    loss = agent.update()

    reward_history.append(total_reward)
    loss_history.append(loss)

    if episode % 100 == 0:
        print(f"episode: {episode}, total reward: {total_reward:.1f}, loss: {loss:.3f}")

env.close()

# ============================================================
# 可視化：報酬推移（学習が進むと伸びるのが期待）
# ============================================================
plot_total_reward(reward_history)

# （任意）損失も見たい場合：方策勾配は損失が単調減少しないことも普通にある点に注意
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# ============================================================
# 評価プレイ（render）
# ============================================================
# 学習済み方策でプレイを確認する。
# Gymnasium の human render は env 作成時に render_mode を指定する必要がある。
eval_env = gym.make("CartPole-v1", render_mode="human")
state, info = eval_env.reset()
done = False
total_reward = 0.0

# 評価時は「確率的にサンプル」だとブレるので、
# greedy（argmax）で行動選択することも多い。
# ただし REINFORCE の方策は確率分布なので、評価方針は目的次第。
with torch.no_grad():
    while not done:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = agent.policy(state_t)  # device=cpu 想定
        action = int(torch.argmax(logits, dim=1).item())

        state, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

eval_env.close()
print("Eval Total Reward:", total_reward)
