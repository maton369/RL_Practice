# ============================================================
# REINFORCE（モンテカルロ方策勾配）の最小例（PyTorch + Gymnasium版）
# ============================================================
# このコードは CartPole（離散行動 2 択）に対して、
# 「確率的方策 π(a|s) をニューラルネットで表し、エピソード全体の報酬で方策を更新する」
# という REINFORCE（Policy Gradient の基本形）を実装している。
#
# 重要な前提：
# - REINFORCE は「エピソード終了後にまとめて更新する」モンテカルロ法（MC）である。
# - TD のようにブートストラップせず、リターン（割引和）G_t を直接使う。
# - 価値関数（V や Q）を別に学習しない最小構成なので、更新は分散が大きくなりやすい。
#   実務では baseline（例：状態価値 V(s)）を入れて分散を下げる Actor-Critic に発展する。
#
# Gym について：
# - Gym は unmaintained なので Gymnasium を使う（NumPy 2.0 周りの互換問題を避けやすい）。
# - Gymnasium の step() は (obs, reward, terminated, truncated, info) を返すので、
#   done = terminated or truncated として「エピソード終了」を統一する。
# ============================================================

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ----------------------------
# 方策ネットワーク πθ(a|s)
# ----------------------------
class Policy(nn.Module):
    """
    確率的方策 πθ(a|s) を表すニューラルネット（PyTorch版）。

    入力：
      - state（CartPole の観測ベクトル：通常 4 次元）

    出力：
      - 各行動の確率（softmax で正規化された分布）
        probs[a] = πθ(a|s)

    なぜ softmax を使うのか：
      - 出力を「確率分布」にし、行動を確率的にサンプルできるようにするため。
      - REINFORCE の更新では log πθ(a_t|s_t) が必要で、
        PyTorch では Categorical(probs) が log_prob を提供してくれるので実装が素直になる。
    """

    def __init__(self, state_dim: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, state_dim)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # 確率に変換（数値安定性の観点では logits を直接 Categorical(logits=...) に渡す手もある）
        probs = F.softmax(logits, dim=-1)
        return probs


@dataclass
class Transition:
    """
    REINFORCE では「エピソード内の行動の log 確率」と「報酬列」が必要になる。

    - log_prob: log πθ(a_t|s_t)
      ※ 後で loss = - Σ_t G_t log_prob_t を作るため
    - reward: r_{t+1}
    """

    log_prob: torch.Tensor
    reward: float


class Agent:
    """
    REINFORCE エージェント（PyTorch版）。

    方策勾配の基本式（モンテカルロ推定）：

      ∇θ J(θ) = E[ Σ_t  G_t ∇θ log πθ(a_t|s_t) ]

    これを「損失の最小化」として書くと：


    $$
    L(\theta) = - \sum_t G_t \log \pi_\theta(a_t|s_t)
    $$


    を最小化することになる。
    つまり、
    - リターン G_t が大きい（良かった）行動は log π を増やす（確率を上げる）
    - リターン G_t が小さい（悪かった）行動は確率を下げる
    方向に更新される。
    """

    def __init__(
        self,
        state_dim: int,
        action_size: int,
        gamma: float = 0.98,
        lr: float = 2e-4,
        device: str | torch.device = "cpu",
        normalize_returns: bool = False,
    ):
        self.gamma = gamma
        self.action_size = action_size
        self.device = torch.device(device)
        self.normalize_returns = normalize_returns

        # 方策ネット πθ
        self.pi = Policy(state_dim=state_dim, action_size=action_size).to(self.device)

        # 最適化：Adam（方策勾配は分散が大きく、Adam が扱いやすいことが多い）
        self.optimizer = torch.optim.Adam(self.pi.parameters(), lr=lr)

        # エピソード内メモリ（オンポリシーなので「エピソードごとに使い切り」）
        self.memory: list[Transition] = []

    def get_action(self, state: np.ndarray) -> int:
        """
        方策 πθ(·|s) から行動をサンプルする。

        実装の要点：
        - Categorical(probs) を使うと、action のサンプルと log_prob を同時に扱える。
        - log_prob を保存しておくと、後で REINFORCE の loss を簡潔に組める。

        注意：
        - state は numpy の観測（shape: (state_dim,)）なので torch Tensor に変換する。
        - バッチ次元を付けて (1, state_dim) とし、ネットへ渡す。
        """
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(
            0
        )  # (1, state_dim)

        # probs: (1, action_size)
        probs = self.pi(state_t).squeeze(0)  # (action_size,)

        # 分布を作り、方策に従って行動をサンプル
        dist = Categorical(probs=probs)
        action_t = dist.sample()  # shape: ()

        # その行動の log 確率（REINFORCE の更新に必要）
        log_prob_t = dist.log_prob(action_t)

        # メモリに保存（reward は step 後に分かるので、ここでは log_prob だけ保持するのではなく、
        # 呼び出し側で reward とセットで add() する設計にしている）
        self._pending_log_prob = log_prob_t  # 直後の add() のために一時保持

        return int(action_t.item())

    def add(self, reward: float):
        """
        直前に選んだ行動の log_prob と、今得た報酬をセットで保存する。

        この保存形式が「REINFORCE の損失」を組み立てる最小情報である：
        - log_prob_t = log πθ(a_t|s_t)
        - reward_{t+1}
        """
        self.memory.append(Transition(log_prob=self._pending_log_prob, reward=reward))

    def reset(self):
        """エピソード開始時にメモリをクリアする。"""
        self.memory.clear()

    def update(self):
        """
        エピソード終了後に 1 回だけ方策更新を行う（モンテカルロ）。

        1) 報酬列から割引リターン G_t を計算（後ろ向きに累積）
           G_t = r_{t+1} + γ r_{t+2} + γ^2 r_{t+3} + ...
        2) loss = - Σ_t G_t * log_prob_t を作る
        3) 逆伝播して optimizer.step()

        分散について：
        - 純粋な REINFORCE は分散が大きい。
        - 代表的な分散低減として、returns の正規化や baseline（Advantage）導入がある。
        - ここではオプションで returns 正規化を切り替え可能にしている（normalize_returns）。
        """
        if len(self.memory) == 0:
            return

        # --- 1) リターン（G_t）を計算 ---
        returns: list[float] = []
        G = 0.0
        for tr in reversed(self.memory):
            G = tr.reward + self.gamma * G
            returns.append(G)
        returns.reverse()  # t=0..T-1 の順に戻す

        # Tensor 化
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 任意：分散低減として正規化（平均0, 分散1）
        # ※ baseline と違って厳密な意味で目的関数を変える可能性はあるが、
        #   教材・小規模タスクでは学習が動きやすくなることがある。
        if self.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # --- 2) 損失を構築 ---
        # REINFORCE の損失：
        #   loss = - Σ_t G_t log πθ(a_t|s_t)
        # memory には log_prob が Tensor として入っているので、そのまま積を取れる
        log_probs_t = torch.stack([tr.log_prob for tr in self.memory])  # (T,)
        loss = -(log_probs_t * returns_t).sum()

        # --- 3) 最適化 ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # オンポリシー：このエピソードのサンプルは使い切り
        self.reset()


# ------------------------------------------------------------
# 学習ループ（Gymnasium API 対応）
# ------------------------------------------------------------
def main():
    # 再現性を少し上げたい場合は seed を固定（完全再現は環境依存もある）
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CartPole は v0 より v1 が推奨
    env = gym.make("CartPole-v1")

    # Gymnasium: reset(seed=...) が使える
    state, info = env.reset(seed=seed)
    state_dim = state.shape[0]
    action_size = env.action_space.n

    agent = Agent(
        state_dim=state_dim,
        action_size=action_size,
        gamma=0.98,
        lr=2e-4,
        device="cpu",
        normalize_returns=False,  # True にすると動きやすくなる場合がある
    )

    episodes = 3000
    reward_history: list[float] = []

    for episode in range(episodes):
        state, info = env.reset()
        agent.reset()

        done = False
        sum_reward = 0.0

        while not done:
            # 方策に従って行動をサンプル
            action = agent.get_action(state)

            # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # 直前行動の log_prob と reward を保存
            agent.add(float(reward))

            state = next_state
            sum_reward += float(reward)

        # エピソード終端後に REINFORCE 更新
        agent.update()

        reward_history.append(sum_reward)
        if episode % 100 == 0:
            print(f"episode :{episode}, total reward : {sum_reward:.1f}")

    # 既存ユーティリティがあるならそれを使い、なければ簡易プロット
    try:
        from common.utils import plot_total_reward

        plot_total_reward(reward_history)
    except Exception:
        import matplotlib.pyplot as plt

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.plot(range(len(reward_history)), reward_history)
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
