import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # 親ディレクトリを import 対象に追加（教材コードでよくある小技）
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class QLearningAgent:
    """
    Q-learning（オフポリシー TD 制御）を最小構成で実装したエージェント。

    このコードがやっていること（アルゴリズムの位置づけ）：
    - 環境モデル（遷移確率や報酬関数）を使わず、経験 (s, a, r, s') だけから学習する model-free 制御。
    - 目標は最適行動価値関数 Q*(s,a) を近似し、最適方策（greedy 方策）に近づけること。

    Q-learning の本質：
    Q-learning は「最適ベルマン方程式」に対応する 1-step ブートストラップ更新を行う。
    次状態 s' で「最良の行動を取る」と仮定した価値でバックアップするため、更新ターゲットに max を使う。

    更新式（1-step）：
        target = r + γ * max_{a'} Q(s', a')
        Q(s,a) <- Q(s,a) + α * (target - Q(s,a))

    - α は学習率
    - γ は割引率
    - (target - Q(s,a)) は TD誤差（Temporal Difference error）

    オフポリシー性：
    - 行動選択は ε-greedy（探索を混ぜる）で行う
    - しかし更新ターゲットは max（greedy）で作る
    つまり「行動は探索込みで取るのに、更新は greedy を仮定する」のでオフポリシーとなる。

    SARSA との対比（理解の補助）：
    - SARSA：target = r + γ Q(s', a')（a' は実際に取る行動）
    - Q-learning：target = r + γ max_{a'} Q(s', a')（最良行動を仮定）
    """

    def __init__(self):
        # 割引率 γ：将来報酬をどれだけ重視するか（0に近いほど短期志向、1に近いほど長期志向）
        self.gamma = 0.9

        # 学習率 α：新しい観測をどれだけ強く反映するか（大きいほど速いが不安定になりやすい）
        self.alpha = 0.8

        # 探索率 ε：ε-greedy における「ランダム行動」の割合
        self.epsilon = 0.1

        # 行動数（GridWorld は 4 方向：UP/DOWN/LEFT/RIGHT）
        self.action_size = 4

        # 行動価値関数 Q(s,a) を辞書で保持
        # 未訪問の (s,a) は 0 で初期化される（optimistic ではない標準初期値）
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        """
        ε-greedy による行動選択。

        - 確率 ε で探索：ランダムに行動を選ぶ
        - 確率 1-ε で活用：現時点の Q 推定が最大の行動を選ぶ

        注意（実装上の癖）：
        - np.argmax は同値最大が複数あるとき「最初のインデックス」を返す。
          初期 Q が全部 0 の間は action=0 に偏りやすい。
          学習の見た目や速度に影響しうるので、ランダムタイブレーク版 argmax を使うことも多い。
        """
        # 探索（Exploration）
        if np.random.rand() < self.epsilon:
            # action_size 個の行動 {0,1,2,3} から一様に選ぶ
            return np.random.choice(self.action_size)

        # 活用（Exploitation）
        # 現状態 s における全行動の Q(s,a) を列挙し、最大の行動を返す
        qs = [self.Q[state, a] for a in range(self.action_size)]
        return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning の TD(0) 更新を行う。

        入力：
        - state: 現状態 s
        - action: 取った行動 a
        - reward: 得た報酬 r (= R_{t+1})
        - next_state: 次状態 s'
        - done: 終端フラグ（エピソード終了か）

        1) 次状態の価値（最大行動価値）を計算：
            max_{a'} Q(s', a')

        2) TDターゲットを構成：
            target = r + γ max_{a'} Q(s', a')

        3) 誤差駆動更新：
            Q(s,a) <- Q(s,a) + α (target - Q(s,a))

        終端の扱い：
        - 終端に到達した場合、将来の価値は存在しないので 0 とみなす。
          よって target = r となり、終端バックアップが行われる。
        """
        # 次状態 s' の最大価値を作る（終端なら 0）
        if done:
            next_q_max = 0
        else:
            # s' における全行動の Q(s',a) を列挙して最大を取る
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # 最適ベルマンバックアップに対応する 1-step ターゲット
        target = reward + self.gamma * next_q_max

        # TD誤差（target - 現推定）で Q を更新
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


# ------------------------------------------------------------
# 実験：GridWorld で Q-learning を実行して Q(s,a) を学習する
# ------------------------------------------------------------
env = GridWorld()
agent = QLearningAgent()

episodes = 1000
for episode in range(episodes):
    # エピソード開始：初期状態へ戻す
    state = env.reset()

    while True:
        # ε-greedy で行動を選ぶ（探索込み）
        action = agent.get_action(state)

        # 環境を 1 ステップ進める
        # next_state: 次状態 s'
        # reward: 報酬 r
        # done: 終端か
        next_state, reward, done = env.step(action)

        # 観測した 1-step 遷移で Q-learning 更新
        agent.update(state, action, reward, next_state, done)

        # 終端ならエピソード終了
        if done:
            break

        # 継続なら次状態へ
        state = next_state

# 学習した Q(s,a) の可視化（各状態・行動の価値を表示）
env.render_q(agent.Q)
