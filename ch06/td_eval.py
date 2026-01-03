import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class TdAgent:
    """
    GridWorld 上で固定方策 π（ここでは一様ランダム）に従って行動しながら、
    TD(0)（Temporal-Difference learning, 1-step TD）で状態価値関数 V(s) を推定するエージェント。

    このコードのアルゴリズム的な位置づけ：
    - 目的：固定方策 π の価値関数 V^π(s) を推定（方策評価）
    - 手法：TD(0) によるオンライン更新（1ステップ先の推定値でブートストラップする）
    - 方策：ランダム方策で固定（改善はしない）→ 「TDによる方策評価」

    MC（モンテカルロ）との対比が重要：
    - MC：エピソード終了まで待って、実際のリターン G を使って更新
    - TD：1ステップ進むたびに、推定値 V(next_state) を使って即座に更新（bootstrap）

    TD(0) の更新式：
    状態 s で報酬 r を得て次状態 s' に遷移したとき、TDターゲットを

        target = r + γ V(s')

    として

        V(s) <- V(s) + α (target - V(s))

    で更新する。
    ここで (target - V(s)) は TD誤差（TD error）と呼ばれ、
    予測誤差を用いた典型的な誤差駆動学習になっている。
    """

    def __init__(self):
        # 割引率 γ：将来報酬の重み（0に近いほど近視眼的、1に近いほど長期志向）
        self.gamma = 0.9

        # 学習率 α：更新の強さ
        # - 大きいと学習が速いが不安定になりやすい
        # - 小さいと安定だが収束が遅い
        self.alpha = 0.01

        # 行動数：GridWorld は 4 行動（UP/DOWN/LEFT/RIGHT）
        self.action_size = 4

        # 固定方策 π：全状態で一様に 0.25 ずつ
        # defaultdict により未登録状態でも同じ分布を返す
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)

        # 推定する価値関数 V(s)：初期値は 0（未訪問でも0を返す）
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        """
        状態 state で方策 π に従って行動をサンプリングする。

        この例では π は一様なので、実質ランダム行動。
        ただし TD学習の観点では「behavior policy が固定」という点が重要で、
        推定対象はこの π の価値 V^π になる。
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        """
        TD(0) による価値更新を 1 ステップ分だけ行う。

        入力：
        - state: 現状態 s
        - reward: 受け取った報酬 r (= R_{t+1})
        - next_state: 次状態 s'
        - done: 終端状態に到達したかどうか

        更新の理論：
        固定方策 π の価値はベルマン期待方程式

            V^π(s) = E[ R_{t+1} + γ V^π(S_{t+1}) | S_t=s ]

        を満たす。TD(0) はこの期待値をサンプルで近似し、
        右辺の V^π(S_{t+1}) を現在の推定値 V(S_{t+1}) で置き換えて更新する。

        TDターゲット：
            target = r + γ V(s')

        ただし終端では次状態の価値を 0 とするのが一般的で、
        このコードでは done=True のとき next_V=0 にしている。
        """
        # 終端状態なら次状態価値は 0 とみなす（エピソードがそこで終了するため）
        next_V = 0 if done else self.V[next_state]

        # 1-step TD ターゲット（ベルマンバックアップのサンプル版）
        target = reward + self.gamma * next_V

        # TD誤差 δ = target - V(s) を用いた誤差駆動更新
        #   V(s) <- V(s) + α δ
        self.V[state] += (target - self.V[state]) * self.alpha


# ------------------------------------------------------------
# 実験ループ：ランダム方策でエピソードを生成しつつ TD(0) で V を推定
# ------------------------------------------------------------
env = GridWorld()
agent = TdAgent()

episodes = 1000
for episode in range(episodes):
    # 環境初期化（開始状態へ）
    state = env.reset()

    while True:
        # 方策に従い行動選択
        action = agent.get_action(state)

        # 環境を1ステップ進める
        next_state, reward, done = env.step(action)

        # 観測した遷移 (s, r, s') で即座に TD 更新する（オンライン更新）
        agent.eval(state, reward, next_state, done)

        # 終端ならエピソード終了
        if done:
            break

        # 続行するなら次状態へ
        state = next_state

# 推定された V(s) を可視化
# ランダム方策の下で、ゴール近傍ほど価値が高くなる傾向が期待される
env.render_v(agent.V)
