import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaAgent:
    """
    SARSA（State-Action-Reward-State-Action）によるオンポリシーTD制御を行うエージェント。

    このコードのアルゴリズム的な位置づけ：
    - 目的：行動価値関数 Q(s,a) を学習し、方策 π を改善して（準）最適方策に近づける（制御）
    - 手法：SARSA（TD(0) のオンポリシー版）を用いた 1-step ブートストラップ更新
    - 方策：ε-greedy 方策 π を維持しつつ、その方策に従って行動（on-policy）

    SARSA の更新式（1-step）：
    状態 s で行動 a を取り、報酬 r を得て次状態 s' に遷移し、
    次に同じ方策 π で選んだ行動を a' とすると、TDターゲットは

        target = r + γ Q(s', a')

    であり、更新は

        Q(s,a) <- Q(s,a) + α (target - Q(s,a))

    となる。ここで
    - (target - Q(s,a)) が TD誤差（TD error）
    - Q(s',a') を用いる点が「オンポリシー」（次行動も実際に取る行動）であることの証拠

    Q-learning との対比：
    - SARSA：target = r + γ Q(s', a') （a' は方策に従ってサンプル）
    - Q-learning：target = r + γ max_a Q(s', a) （greedy でブートストラップ、オフポリシー）

    本実装の工夫：
    - SARSA は更新に (s,a,r,s',a') が必要なので、直前と次のステップの情報を2個ぶん保持する。
    - そのために deque(maxlen=2) を使い、「2つ揃ったら1回更新」する設計にしている。
    """

    def __init__(self):
        # 割引率 γ：将来報酬の重み
        self.gamma = 0.9

        # 学習率 α：更新の強さ（ここでは 0.8 と大きめで、学習が速い反面ブレやすい）
        self.alpha = 0.8

        # 探索率 ε：ε-greedy 方策で探索を残す割合
        self.epsilon = 0.1

        # 行動数（UP/DOWN/LEFT/RIGHT）
        self.action_size = 4

        # 方策 π の初期化：一様ランダム
        # 学習開始直後の Q が未熟な段階でも探索を確保できる。
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)

        # 行動価値関数 Q(s,a)：未定義の (s,a) は 0
        self.Q = defaultdict(lambda: 0)

        # SARSA 更新のために「2ステップ分の情報」を保持するバッファ
        # memory[0] = (s, a, r, done)
        # memory[1] = (s', a', r', done') になる想定
        # maxlen=2 なので常に直近2つだけが残る
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        """
        現在の方策 π(·|state) に従って行動をサンプリングする。

        SARSA はオンポリシーなので、ここで選んだ行動が
        次の更新式の a'（次行動）としても使われることになる。
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        """
        エピソード開始時にバッファをリセットする。

        バッファは (s,a,r,done) を2つ貯めて更新する方式なので、
        エピソード間で混ざらないようにクリアが必要。
        """
        self.memory.clear()

    def update(self, state, action, reward, done):
        """
        1ステップ分の遷移情報を受け取り、2つ揃ったら SARSA 更新を1回行う。

        入力：
        - state: 現状態 s
        - action: 取った行動 a
        - reward: 得た報酬 r (= R_{t+1})
        - done: 終端かどうか

        実装戦略：
        SARSA のターゲットには Q(s', a') が必要であるため、
        その場ではまだ a' が確定していないタイミングが存在する。
        そこで
        - 現在の (s,a,r,done) を memory に入れる
        - 次ステップで (s',a',r',done') が入った時点で
          memory[0] を更新対象として、memory[1] から (s',a') を参照して更新する

        これにより「行動選択→環境ステップ→次行動選択→更新」の依存関係を
        シンプルに処理できる。
        """
        # 現在の遷移情報を保存
        self.memory.append((state, action, reward, done))

        # まだ2つ揃っていない（初回の1ステップ目など）なら更新できないので待つ
        if len(self.memory) < 2:
            return

        # 更新対象：1つ前の遷移（s,a,r,done）
        state, action, reward, done = self.memory[0]

        # 次ステップ情報（s', a'）
        next_state, next_action, _, _ = self.memory[1]

        # 終端なら次状態価値は 0 とみなす（episodic設定）
        # ※ done は「更新対象の遷移の結果として終端に到達したか」を表す。
        next_q = 0 if done else self.Q[next_state, next_action]

        # SARSA の TDターゲット：r + γ Q(s',a')
        target = reward + self.gamma * next_q

        # TD誤差に基づく誤差駆動更新：
        #   Q(s,a) <- Q(s,a) + α (target - Q(s,a))
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策改善：更新後の Q を使って、その状態の方策を ε-greedy に更新
        # greedy_probs は
        # - greedy 行動に 1-ε + ε/|A|
        # - それ以外に ε/|A|
        # を割り当てる分布を返す想定。
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


# ------------------------------------------------------------
# 実験ループ：SARSA（オンポリシーTD制御）で Q と π を学習
# ------------------------------------------------------------
env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    # エピソード開始：環境を初期化
    state = env.reset()

    # 2ステップバッファをクリア
    agent.reset()

    while True:
        # 現方策に従って行動を選択（ε-greedy のサンプル）
        action = agent.get_action(state)

        # 環境を1ステップ進める（次状態・報酬・終端フラグを得る）
        next_state, reward, done = env.step(action)

        # この時点では next_action がまだ決まっていないので、
        # update は「情報を貯める」だけで、次の update 呼び出し時に更新が走る。
        agent.update(state, action, reward, done)

        if done:
            # エピソード終了時の注意：
            # SARSA は (s,a,r,s',a') が必要なので、
            # 終端に到達した最後の遷移は「次の (s',a')」が通常のループでは供給されない。
            #
            # そこで、ここではダミーの呼び出しを1回入れて
            # memory を2個揃え、最後の遷移に対する更新を発火させている。
            #
            # この呼び出しで memory[1] に入る next_state は終端状態であり、
            # update 内では done=True なら next_q=0 になるため、
            # 実質的に target = reward で終端バックアップが行われる。
            agent.update(next_state, None, None, None)
            break

        # 継続なら次状態へ
        state = next_state

# 学習した Q(s,a) を可視化
env.render_q(agent.Q)
