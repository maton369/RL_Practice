import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class QLearningAgent:
    """
    Q-learning によるオフポリシー TD 制御（model-free control）を行うエージェント。

    このコードのアルゴリズム的な位置づけ：
    - 目的：最適行動価値関数 Q*(s,a) を学習し、そこから最適方策（greedy 方策）を得る（制御）
    - 手法：Q-learning（TD(0) のオフポリシー版）を用いた 1-step ブートストラップ更新
    - データ生成：探索用の行動方策 b（ε-greedy）で行動しつつ、
      更新ターゲットは greedy（max）で作るため「オフポリシー」になる。

    Q-learning のコア更新式：
    状態 s で行動 a を取り、報酬 r を得て次状態 s' に遷移したとき

        target = r + γ max_{a'} Q(s', a')

        Q(s,a) <- Q(s,a) + α (target - Q(s,a))

    となる。
    - max を使う点が「次に実際に取る行動」ではなく「最良行動」を仮定して更新する、という意味でオフポリシー
    - この max によるブートストラップが、最適ベルマン作用素（Bellman optimality operator）の固定点に近づける

    SARSA との対比：
    - SARSA（オンポリシー）: target = r + γ Q(s', a') （a' は π でサンプルした実行動）
    - Q-learning（オフポリシー）: target = r + γ max_a Q(s', a) （greedy を仮定）

    本実装の設計：
    - pi：学習した Q から作る greedy 方策（ε=0）
    - b ：実際に行動してデータを集める ε-greedy 方策（探索を残す）
      → 行動は b、更新ターゲットは max（=pi 的）という分離でオフポリシー性が明確
    """

    def __init__(self):
        # 割引率 γ（将来報酬の重み）
        self.gamma = 0.9

        # 学習率 α（更新の強さ。大きいと速いが不安定になりやすい）
        self.alpha = 0.8

        # 探索率 ε（行動方策 b の ε-greedy に用いる）
        self.epsilon = 0.1

        # 行動数（GridWorld: UP/DOWN/LEFT/RIGHT）
        self.action_size = 4

        # 初期方策は一様ランダム
        # - pi は後で greedy（ε=0）に更新し「改善対象」にする
        # - b  は ε-greedy に更新し「探索用」にする
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(
            lambda: random_actions
        )  # target policy（参考用。実際の更新ターゲットは max で表現）
        self.b = defaultdict(
            lambda: random_actions
        )  # behavior policy（実際に行動する方策）

        # 行動価値関数 Q(s,a)（未知は 0 から開始）
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        """
        行動は行動方策 b に従ってサンプルする（探索用）。

        b を ε-greedy にしておくことで
        - 状態・行動を幅広く試す（探索）
        - b(a|s) > 0 が確保されやすい（オフポリシー学習の前提として重要）
        という性質が得られる。
        """
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning の 1-step TD 更新を行う。

        入力：
        - state: 現状態 s
        - action: 取った行動 a
        - reward: 得た報酬 r (= R_{t+1})
        - next_state: 次状態 s'
        - done: 終端かどうか

        1) 次状態の最大行動価値を計算
            max_{a'} Q(s',a')

        2) TDターゲットを作る
            target = r + γ max_{a'} Q(s',a')

        3) 誤差駆動更新
            Q(s,a) <- Q(s,a) + α (target - Q(s,a))

        終端の扱い：
        - episodic 環境では終端後の価値は 0 とみなすため
            max_{a'} Q(terminal, a') = 0
          として target = r になる（終端バックアップ）。
        """
        if done:
            # 終端では次状態の価値を 0 とする（これ以上報酬が続かない）
            next_q_max = 0
        else:
            # 次状態 s' における全行動の Q を列挙し、その最大値を取る
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # 最適ベルマン方程式に基づく 1-step ターゲット
        target = reward + self.gamma * next_q_max

        # TD誤差に基づく更新（誤差駆動）
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策の更新（GPI の「改善」部分）
        # - pi: greedy（ε=0）→ 学習した Q に対する最良方策（ターゲット方策の参照）
        # - b : ε-greedy       → 実際の探索行動方策
        #
        # 注意：Q-learning の更新式自体は max を使うので、pi を明示的に持たなくても成立する。
        # ここでは「学習したい方策（pi）と行動方策（b）の分離」を明示するために両方更新している。
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


# ------------------------------------------------------------
# 学習ループ：b で行動しながら Q-learning 更新で Q を最適化する
# ------------------------------------------------------------
env = GridWorld()
agent = QLearningAgent()

episodes = 10000
for episode in range(episodes):
    # エピソード開始：環境を初期化
    state = env.reset()

    while True:
        # 行動は探索用方策 b から選ぶ
        action = agent.get_action(state)

        # 環境遷移（次状態・報酬・終端フラグ）
        next_state, reward, done = env.step(action)

        # Q-learning 更新（ターゲットは greedy max を使うのでオフポリシー）
        agent.update(state, action, reward, next_state, done)

        # 終端ならエピソード終了
        if done:
            break

        # 継続なら次状態へ
        state = next_state

# 学習した Q(s,a) を可視化
env.render_q(agent.Q)
