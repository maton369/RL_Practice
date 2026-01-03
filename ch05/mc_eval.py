import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class RandomAgent:
    """
    ランダム方策で GridWorld をプレイし、その経験（エピソード）から
    モンテカルロ法（Monte Carlo; MC）で状態価値関数 V(s) を推定するエージェント。

    このコードでやっていること（強化学習的な位置づけ）：
    - 方策 π は固定（ここでは一様ランダム）で、改善はしない。
    - エピソードを何回も生成し、各状態の「リターン（return）」を平均して
      V^π(s) を推定する（方策評価）。
    - したがってアルゴリズムは「MCによる方策評価（Monte Carlo policy evaluation）」である。

    MC方策評価の基本：
    - ある状態 s を訪れたとき、その後に得る割引報酬和（リターン）を

      $$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots $$

      と定義する。
    - V^π(s) は「方策 π に従ったときの G の期待値」なので

      $$ V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s] $$

      をサンプル平均で近似する。

    注意：
    - この実装は「状態 s を訪れた回数 cnts[s] で平均を取る」ので、
      サンプル平均型のMC推定になっている。
    - ただし「同一エピソード内で同じ状態を複数回訪れた場合」も全部カウントするため、
      これは厳密には "every-visit Monte Carlo"（毎訪問MC）に相当する。
      "first-visit MC"（初回訪問MC）にしたい場合は、エピソード内で最初の1回だけ更新する必要がある。
    """

    def __init__(self):
        # 割引率 γ（将来報酬をどれだけ重視するか）
        self.gamma = 0.9

        # 行動数（GridWorld: UP/DOWN/LEFT/RIGHT の4つ）
        self.action_size = 4

        # 一様ランダム方策 π(a|s) を定義：
        # 全状態で同じ分布を返す（0.25ずつ）
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}

        # pi[state] -> {action: prob}
        # defaultdict を使うことで、未登録 state でも常に一様分布が返る
        self.pi = defaultdict(lambda: random_actions)

        # V[state] -> 推定価値 V(s)
        # 初期値 0 は任意（試行を重ねると期待値へ近づく）
        self.V = defaultdict(lambda: 0)

        # cnts[state] -> 状態 state を訪れた回数（every-visit のカウント）
        # サンプル平均更新で必要になる
        self.cnts = defaultdict(lambda: 0)

        # memory: 1エピソード分の遷移（state, action, reward）列を保存
        # MCはエピソードの終わりまで待ってから更新するため、
        # こうして一旦ためておく必要がある（TDとの違いの重要ポイント）
        self.memory = []

    def get_action(self, state):
        """
        状態 state において方策 π に従って行動をサンプリングする。

        実装：
        - action_probs = π(·|state) を取得
        - np.random.choice(actions, p=probs) で確率的に行動を選ぶ

        この例では π は常に一様なので、実質「ランダム行動」になる。
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """
        エピソード中に観測した (state, action, reward) を memory に追加する。

        MC法では「将来の報酬を含めたリターン G」を計算してから価値を更新したいので、
        エピソードが終わるまで遷移列を保存しておく必要がある。
        """
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        """
        エピソード開始時に memory をクリアする。

        1エピソード分の経験のみを保持し、
        エピソード終了時（done=True）に MC 更新を行う。
        """
        self.memory.clear()

    def eval(self):
        """
        1エピソード分の memory を使って V(s) をモンテカルロ更新する。

        手順：
        - エピソード終端から逆向きにたどりながらリターン G を計算する。
        - 各 state について、そのときの G をサンプルとして V(state) の平均との差分更新を行う。

        逆向きに計算する理由：
        - リターンは将来報酬の割引和なので、末尾からなら

            G <- γ G + reward

          の1行で更新できる（効率的）。

        更新式（サンプル平均）：
        状態 s を n 回観測し、観測リターンを G_1,...,G_n とすると

            V_n(s) = (1/n) Σ_i G_i

        逐次更新形は

            V(s) <- V(s) + (G - V(s)) / n

        であり、これを実装している。
        """
        G = 0  # エピソード末尾からの累積割引報酬（リターン）

        # エピソードの最後から最初へ逆順で走査
        for data in reversed(self.memory):
            state, action, reward = data

            # 逆方向のリターン更新：
            #   G_t = R_{t+1} + γ G_{t+1}
            G = self.gamma * G + reward

            # every-visit MC：この状態が出てくるたびにカウントして更新する
            self.cnts[state] += 1

            # サンプル平均で V(s) を更新（学習率は 1/cnts[state]）
            self.V[state] += (G - self.V[state]) / self.cnts[state]


# ------------------------------------------------------------
# 実験ループ：ランダム方策でエピソードを生成し、MCで V を推定する
# ------------------------------------------------------------
env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    # 環境を初期状態へ
    state = env.reset()

    # エピソード経験のバッファを初期化
    agent.reset()

    while True:
        # 方策に従って行動選択（ここではランダム）
        action = agent.get_action(state)

        # 環境を1ステップ進める
        next_state, reward, done = env.step(action)

        # この時刻の (state, action, reward) を記録
        # 注意：ここで保存している reward は「次状態に遷移した結果として得た報酬」であり、
        # リターン計算の定義（R_{t+1}）と整合する。
        agent.add(state, action, reward)

        # 終端に到達したら、そのエピソード全体を使ってMC更新
        if done:
            agent.eval()
            break

        # 次状態へ遷移して継続
        state = next_state

# 推定された V(s) を描画（ランダム方策の下での状態価値）
env.render_v(agent.V)
