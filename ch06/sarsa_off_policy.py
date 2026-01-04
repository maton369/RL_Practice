if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaOffPolicyAgent:
    """
    オフポリシー版の SARSA 風更新（Importance Sampling を組み込んだ TD 制御）のエージェント。

    まず前提整理：
    - オンポリシー（SARSA）では、行動を選ぶ方策と「評価・改善したい方策」が同一である。
      そのとき更新ターゲットは
          r + γ Q(s', a')
      で、a' は実際に次に取る行動（πからサンプル）である。

    - オフポリシーでは
      - データを集める方策（行動方策 / behavior policy）を b
      - 評価・改善したい方策（ターゲット方策 / target policy）を π
      と分ける。
      行動は b に従うが、推定したいのは π の下での価値（または最適方策）である。

    このコードの狙い：
    - 行動は探索込みの b（ε-greedy）で生成して探索を担保しつつ、
    - ターゲット方策 π を greedy（ε=0）にして「改善対象は greedy 方策」として扱い、
    - オフポリシー補正として importance ratio ρ を TD ターゲットに掛けて更新する。

    重要度比（importance ratio）の基本：
    オフポリシーでは、b で生成したサンプルを π の期待値として扱うために
        ρ = π(a|s) / b(a|s)
    を用いて補正する。

    注意（理論的に重要）：
    - ここで実装している ρ は「次状態 next_state における next_action の比」
      つまり ρ = π(a'|s') / b(a'|s') になっている。
    - 一般的なオフポリシー TD（例：Expected SARSA や Tree-backup、Q-learning 等）では
      目標の立て方・比の掛け方に流儀があり、分散や安定性が大きく変わる。
      本コードは教材として「比を掛けるとオフポリシー補正になる」形を示しているが、
      厳密には「どの比をどこに掛けるか」がアルゴリズム定義そのものになる点に注意。

    それでも学習の直観は次の通り：
    - π を greedy に近づけたい（改善）
    - ただし greedy だけだと探索不足になりうるので b を ε-greedy にして探索させる
    - b で集めたデータを π の評価に流用するために ρ で補正する
    """

    def __init__(self):
        # 割引率 γ
        self.gamma = 0.9

        # 学習率 α（大きめ。学習は速いが揺れやすい）
        self.alpha = 0.8

        # 行動方策 b の探索率 ε
        self.epsilon = 0.1

        # 行動数（GridWorld は 4）
        self.action_size = 4

        # 初期方策：一様ランダム
        # - π（ターゲット方策）：後で greedy（ε=0）に更新していく
        # - b（行動方策）：後で ε-greedy に更新して探索を残す
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # target policy π
        self.b = defaultdict(lambda: random_actions)  # behavior policy b

        # 行動価値関数 Q(s,a)（初期は0）
        self.Q = defaultdict(lambda: 0)

        # 2ステップ分の遷移を保持（SARSA は (s,a,r,s',a') が必要）
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        """
        行動選択は行動方策 b に従う（オフポリシーなのでここが重要）。

        - b は ε-greedy なので、学習中も探索が残り、状態空間をカバーしやすい。
        - 一方で π は greedy（ε=0）へ寄せていくため、
          「探索で集めたデータを、より良い方策の学習に使う」構造になる。
        """
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        """エピソード開始時に 2-step バッファをクリアする。"""
        self.memory.clear()

    def update(self, state, action, reward, done):
        """
        1ステップ分のデータを受け取り、2つ揃ったら更新を1回行う。

        バッファ戦略：
        - (s,a,r,done) を memory に積む
        - 次回呼び出しで (s',a',...) が積まれたら、前の遷移を更新する

        更新の中心：
        - next_state で next_action を取ったときの Q(next_state, next_action) を bootstrap に使う
        - そのサンプルが π の下でどれだけ起こりやすいかを ρ で補正する

        重要度比：
            ρ = π(a'|s') / b(a'|s')

        これにより「bで生成した next_action を π の視点で再重み付けする」効果がある。
        """
        # 現ステップの遷移情報を保存
        self.memory.append((state, action, reward, done))

        # 2つ揃わないと (s',a') が参照できないので更新できない
        if len(self.memory) < 2:
            return

        # 更新対象（1つ前の遷移）
        state, action, reward, done = self.memory[0]

        # 次ステップ側の情報（s', a'）
        next_state, next_action, _, _ = self.memory[1]

        # 終端到達なら、次状態価値は 0。
        # また、比を掛ける意味も薄いので ρ=1 としている（終端の安全な扱い）。
        if done:
            next_q = 0
            rho = 1
        else:
            # オフポリシー bootstrap：b が選んだ a' に対して Q(s',a') を参照
            next_q = self.Q[next_state, next_action]

            # 重要度比 ρ = π(a'|s') / b(a'|s')
            # 注意：b(a'|s') が 0 だと破綻するので、
            # 実験上は b が全行動に非ゼロ確率を持つ（ε>0）ことが大事。
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        # TDターゲット（ここでは ρ をターゲット全体に掛ける形）
        #   target = ρ * (r + γ Q(s',a'))
        #
        # 一般論として importance sampling は分散が増えやすい。
        # 特に π が greedy（確率0/1）に近づくと、
        # π(a'|s') が 0 になる行動が出てきて ρ=0 で更新が消えるケースが増える。
        # これは「π では起こらない行動を b が取ったとき、そのサンプルは π の評価に寄与しない」
        # という解釈に対応する。
        target = rho * (reward + self.gamma * next_q)

        # 誤差駆動更新：
        #   Q(s,a) <- Q(s,a) + α (target - Q(s,a))
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 方策改善：
        # - π は greedy（ε=0）に更新 → 目標は greedy 方策
        # - b は ε-greedy に更新 → 探索を維持してデータ収集
        #
        # ここが「探索用の b と、改善対象の π を分離する」オフポリシー構造の要点。
        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


# ------------------------------------------------------------
# 学習ループ：b で行動し、π を改善し、IS で補正しながら Q を学習
# ------------------------------------------------------------
env = GridWorld()
agent = SarsaOffPolicyAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        # 行動は b から選ぶ（オフポリシー）
        action = agent.get_action(state)

        # 環境遷移
        next_state, reward, done = env.step(action)

        # バッファに積む（次の a' が揃ったタイミングで更新が走る）
        agent.update(state, action, reward, done)

        if done:
            # SARSA系は (s',a') が必要なので、終端後にダミー呼び出しで最終更新を発火させる
            agent.update(next_state, None, None, None)
            break

        state = next_state

# 学習した Q を可視化
env.render_q(agent.Q)
