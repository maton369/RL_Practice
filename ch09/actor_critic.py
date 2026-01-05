# ============================================================
# 1-step Actor-Critic（A2Cの最小形）の実装（PyTorch + Gymnasium版）
# ============================================================
# 元コード（DeZero + Gym）でやっていた内容は、強化学習でいうところの
# 「Actor-Critic（方策ネット + 価値ネット）」です。
#
# - Actor（PolicyNet） : 確率的方策  πθ(a|s)  を学習する（行動を出す側）
# - Critic（ValueNet） : 状態価値    Vw(s)    を学習する（評価する側）
#
# REINFORCE（純粋なモンテカルロ方策勾配）ではエピソード全体のリターン G を使って
# 方策を更新しますが、Actor-Critic では Critic が推定する V(s) を使って
# 「その場で（オンラインに）学習」できるのがポイントです。
#
# 1-step の更新（TD(0)）は次の形です。
#
# TDターゲット（ブートストラップ）：
#
# $$
# y_t = r_{t+1} + \gamma (1-\text{done}) V_w(s_{t+1})
# $$
#
# TD誤差（= advantage の最小形）：
#
# $$
# \delta_t = y_t - V_w(s_t)
# $$
#
# Critic（価値ネット）の損失：
#
# $$
# L_V(w) = (V_w(s_t) - y_t)^2
# $$
#
# Actor（方策ネット）の損失（方策勾配）：
#
# $$
# L_\pi(\theta) = -\log \pi_\theta(a_t|s_t)\,\delta_t
# $$
#
# 直感：
# - δ が正（思ったより良い）なら、その行動の確率を上げる方向に更新する
# - δ が負（思ったより悪い）なら、その行動の確率を下げる方向に更新する
#
# 重要：y_t や δ_t を作るとき、ターゲット側（V(s_{t+1})）へ勾配を流すと
# 目的がブレて不安定になりやすいので、ターゲットは detach（no_grad）で固定します。
# これは TD 学習でよく出てくる semi-gradient の考え方です。
#
# Gym について：
# - Gym は unmaintained なので Gymnasium を使います（NumPy 2.0 互換問題を避けやすい）
# - CartPole は v0 より v1 が推奨です
# - Gymnasium の API は reset/step の戻り値が Gym と少し違います
# ============================================================

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Gym は Gymnasium を推奨（Gym の警告/互換問題回避）
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ------------------------------------------------------------
# PolicyNet（Actor）：πθ(a|s) を表すネットワーク
# ------------------------------------------------------------
class PolicyNet(nn.Module):
    """
    確率的方策 πθ(a|s) を表すニューラルネットです。

    実装の定石として、ここでは「確率（softmax結果）」そのものではなく
    logits（softmax 前のスコア）を出力します。

    理由：
    - Categorical(logits=...) を使うと数値的に安定しやすい
      （確率が 0 に近づいて log が発散するのを避けやすい）
    - log πθ(a|s) を取る処理も distribution が安全にやってくれる
    """

    def __init__(self, state_dim=4, action_size=2):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        # x: shape (batch, state_dim)
        x = F.relu(self.l1(x))
        logits = self.l2(x)  # softmax 前のスコア
        return logits


# ------------------------------------------------------------
# ValueNet（Critic）：Vw(s) を表すネットワーク
# ------------------------------------------------------------
class ValueNet(nn.Module):
    """
    状態価値 Vw(s) を近似するネットワークです。
    出力はスカラー（shape (batch, 1)）になります。
    """

    def __init__(self, state_dim=4):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        v = self.l2(x)
        return v


# ------------------------------------------------------------
# Agent：1-step Actor-Critic
# ------------------------------------------------------------
class Agent:
    """
    1-step Actor-Critic（TD(0)）を行うエージェントです。

    - 行動は Actor の確率分布からサンプル（オンポリシー）
    - 学習は各ステップで更新（モンテカルロではなく TD）
    - Critic が作る TD 誤差 δ を advantage の役割として Actor に渡す
    """

    def __init__(self, state_dim=4, action_size=2, device=None):
        # 割引率 γ
        self.gamma = 0.98

        # 学習率：Actor と Critic は役割が違うので別にするのが一般的です
        self.lr_pi = 2e-4
        self.lr_v = 5e-4

        self.action_size = action_size

        # デバイス（GPUがあれば使えるようにする）
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ネットワーク
        self.pi = PolicyNet(state_dim=state_dim, action_size=action_size).to(
            self.device
        )
        self.v = ValueNet(state_dim=state_dim).to(self.device)

        # 最適化（Adam は分散の大きい勾配でも扱いやすいことが多いです）
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state_np):
        """
        状態 s を受け取り、方策 πθ(·|s) から行動をサンプルして返します。

        返すもの：
        - action: int（0 or 1）
        - log_prob: log πθ(a|s)（学習に使う）
        """
        # NumPy -> Torch（batch 次元も付ける）
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # logits -> 分布
        logits = self.pi(state)
        dist = Categorical(logits=logits)

        # 行動サンプル（オンポリシー）
        action = dist.sample()

        # 方策勾配に必要な log πθ(a|s)
        log_prob = dist.log_prob(action)

        return int(action.item()), log_prob

    def update(self, state_np, log_prob, reward, next_state_np, done):
        """
        1ステップぶんの遷移 (s, a, r, s', done) で Actor/Critic を更新します。

        Critic：
          y = r + γ(1-done)V(s')
          L_V = (V(s) - y)^2

        Actor：
          δ = y - V(s)
          L_π = -logπ(a|s) * δ

        重要：
        - y はターゲットなので detach（no_grad）で固定します
        - δ も Actor 側では detach して「重み」としてだけ使います
        """
        # NumPy -> Torch（batch 次元あり）
        state = torch.as_tensor(
            state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_state = torch.as_tensor(
            next_state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # done は bool -> 0/1 のマスクにする（終端なら将来価値を足さない）
        done_f = 1.0 if done else 0.0
        done_mask = torch.tensor(
            [done_f], dtype=torch.float32, device=self.device
        ).unsqueeze(
            1
        )  # shape (1,1)

        # 現在の価値推定 V(s)
        v = self.v(state)  # shape (1,1)

        # 次状態価値 V(s') を使って TD ターゲットを作る（ターゲット側は勾配停止）
        with torch.no_grad():
            next_v = self.v(next_state)  # shape (1,1)
            # y = r + γ(1-done)V(s')
            target = torch.tensor(
                [[reward]], dtype=torch.float32, device=self.device
            ) + self.gamma * next_v * (1.0 - done_mask)

        # TD誤差 δ = y - V(s)
        # Actor では advantage 的な重みとして使うので detach します
        delta = (target - v).detach()

        # --- Critic loss（価値回帰） ---
        # L_V = (V(s) - y)^2
        loss_v = F.mse_loss(v, target)

        # --- Actor loss（方策勾配） ---
        # L_π = -logπ(a|s) * δ
        # log_prob は shape (1,) なので shape を合わせるために unsqueeze(1)
        loss_pi = -(log_prob.unsqueeze(1) * delta).mean()

        # まず勾配をクリア
        self.opt_v.zero_grad()
        self.opt_pi.zero_grad()

        # 逆伝播（2つのネットは別なので別々に backward してOK）
        loss_v.backward()
        loss_pi.backward()

        # パラメータ更新
        self.opt_v.step()
        self.opt_pi.step()


# ------------------------------------------------------------
# 学習ループ（Gymnasium API）
# ------------------------------------------------------------
episodes = 3000
env = gym.make("CartPole-v1")

agent = Agent(state_dim=env.observation_space.shape[0], action_size=env.action_space.n)
reward_history = []

for episode in range(episodes):
    # Gymnasium: reset() -> (obs, info)
    state, info = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        # 方策から行動をサンプル
        action, log_prob = agent.get_action(state)

        # Gymnasium: step() -> (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        # 終端判定
        done = bool(terminated or truncated)

        # 1-step Actor-Critic 更新（オンライン）
        agent.update(state, log_prob, reward, next_state, done)

        # 遷移
        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print(f"episode : {episode}, total reward : {total_reward:.1f}")

# plot（common.utils がある前提）
from common.utils import plot_total_reward

plot_total_reward(reward_history)
