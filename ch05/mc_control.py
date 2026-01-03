import os, sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # for importing the parent dirs
import numpy as np
from collections import defaultdict
from common.gridworld import GridWorld

# from common.utils import greedy_probs


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """
    ε-greedy 方策の「確率分布 π(·|state)」を Q から生成するヘルパー関数。

    入力：
    - Q: 行動価値関数 Q(s,a) を (state, action) -> value で保持する辞書
    - state: 方策を作りたい状態 s
    - epsilon: 探索率 ε（確率 ε でランダム行動、確率 1-ε で greedy 行動）
    - action_size: 行動数 |A|

    出力：
    - action_probs: dict {action: prob} で表した確率分布

    理論：
    ε-greedy は、greedy 行動 a* = argmax_a Q(s,a) を基本としつつ、
    非ゼロ確率で探索を残すことで「全行動が選ばれる可能性」を確保する。

    具体的には：
    - 各行動にまず一様に ε/|A| を配る（探索成分）
    - greedy 行動 a* に追加で (1-ε) を乗せる（活用成分）

    よって

        π(a|s) =
          1-ε + ε/|A|   (a = a*)
          ε/|A|         (otherwise)

    となる。

    注意（タイブレーク）：
    - np.argmax は最大値が複数あるとき「最初の index」を返す。
      同率最大の扱いは実装依存になるため、結果の方策が変わる可能性がある。
    """
    # 状態 state における各行動の Q(s,a) を配列として取り出す
    qs = [Q[(state, action)] for action in range(action_size)]

    # greedy 行動 a* を求める
    max_action = np.argmax(qs)

    # 探索成分：全行動に ε/|A| を配る
    base_prob = epsilon / action_size
    action_probs = {
        action: base_prob for action in range(action_size)
    }  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}

    # 活用成分：(1-ε) を greedy 行動へ追加する
    action_probs[max_action] += 1 - epsilon

    return action_probs


class McAgent:
    """
    モンテカルロ制御（Monte Carlo Control）の簡易実装。
    GridWorld 上で「ε-greedy 方策」を学習し、行動価値関数 Q(s,a) を推定する。

    このコードのアルゴリズム的な位置づけ：
    - 価値推定：エピソード全体からリターン G を計算し、Q(s,a) を更新（MC）
    - 方策改善：更新後の Q に基づき、各状態の方策 π(·|s) を ε-greedy に更新
    - これを繰り返して、最適方策 π* に近づける（制御：control）

    重要な概念：
    - MC（モンテカルロ）は「エピソードが終わるまで待ってから」リターンを計算し更新する。
      （TDのように1ステップで bootstrap しない）
    - ε-greedy にすることで探索を維持し、最適行動に収束しやすくする。
    - 更新はサンプル平均ではなく「定数ステップサイズ α」で行っているため、
      これは

        Q <- Q + α (G - Q)

      という誤差駆動更新（指数移動平均）になる。
      サンプル平均（α=1/N）よりも実装が簡単で、非定常性にもある程度追従可能。

    注意：
    - この実装は「同一エピソード内で同じ (s,a) を複数回更新する」ので
      every-visit MC 的な挙動になりやすい。
    - さらに、エピソード生成方策（behavior policy）と更新後方策（target policy）が同じ
      オンポリシー（on-policy）MC制御に対応している（ε-greedy のままサンプルする）。
    """

    def __init__(self):
        # 割引率 γ（将来報酬の重み）
        self.gamma = 0.9

        # 探索率 ε（ε-greedy の探索強度）
        self.epsilon = 0.1

        # 学習率 α（定数ステップサイズ）
        # - 大きい：新しいサンプルを強く反映（追従性↑、分散↑）
        # - 小さい：推定が安定（分散↓、追従性↓）
        self.alpha = 0.1

        # 行動数（UP/DOWN/LEFT/RIGHT）
        self.action_size = 4

        # 方策 π(·|s) の初期化：一様ランダム
        # 初期Qが全部0の状態だと greedy が未定義になりやすいので、
        # 最初はランダムに動いて経験を集める意味でも妥当。
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)

        # Q(s,a) の推定値：未定義の (s,a) は 0
        self.Q = defaultdict(lambda: 0)

        # 1エピソード分の経験 (state, action, reward) を蓄えるバッファ
        # MCなのでエピソード終了時にまとめて更新するために必要。
        self.memory = []

    def get_action(self, state):
        """
        現在の方策 π(·|state) から行動をサンプリングする。

        ここでサンプルされる方策は常に「最新の ε-greedy 方策」なので、
        アルゴリズムはオンポリシー（on-policy）になっている。
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        """
        1ステップの経験 (S_t, A_t, R_{t+1}) を memory に保存する。

        MCはリターン G_t を計算するために「将来の報酬列」が必要になるので、
        エピソード終了まで蓄積しておく。
        """
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        """
        エピソード開始時に memory をクリアする。
        """
        self.memory.clear()

    def update(self):
        """
        1エピソード分の経験から、Q と π を更新する（MC制御のコア）。

        手順：
        1) エピソード終端から逆向きにリターン G を計算：
              G <- R_{t+1} + γ G
        2) 各 (s,a) を MC ターゲット G で更新：
              Q(s,a) <- Q(s,a) + α (G - Q(s,a))
        3) 更新後の Q を使って、その状態の方策 π(·|s) を ε-greedy に改善する。

        理論的には「Generalized Policy Iteration（GPI）」の形で、
        - 推定（policy evaluation の近似）
        - 改善（policy improvement）
        を交互に行っている。
        """
        G = 0  # 逆順で計算する割引報酬和（リターン）

        # エピソードの最後から最初へ
        for data in reversed(self.memory):
            state, action, reward = data

            # リターン更新：G_t = R_{t+1} + γ G_{t+1}
            G = self.gamma * G + reward

            # (state, action) をキーとして Q(s,a) を更新
            key = (state, action)

            # 定数ステップサイズ α の誤差駆動更新
            # TD(0) の形と似ているが、ターゲットが bootstrap ではなく「実リターン G」なのが MC。
            self.Q[key] += (G - self.Q[key]) * self.alpha

            # 方策改善：更新した Q をもとに、その状態の ε-greedy 方策を再計算して上書きする
            # ※ ここでは「状態ごと」に方策分布を更新している点に注目。
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


# ------------------------------------------------------------
# 実験ループ：MC制御で Q と π を学習する
# ------------------------------------------------------------
env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    # 環境を初期化（開始状態へ）
    state = env.reset()

    # エピソード経験をリセット
    agent.reset()

    while True:
        # 現在方策（ε-greedy）に従って行動選択
        action = agent.get_action(state)

        # 1ステップ進めて遷移・報酬を観測
        next_state, reward, done = env.step(action)

        # 経験を保存（MCのためエピソード全体が必要）
        agent.add(state, action, reward)

        # エピソード終了（ゴール到達）でまとめて更新
        if done:
            agent.update()
            break

        # 継続する場合は次状態へ
        state = next_state

# 学習した Q(s,a) を可視化する
# render_q は各セルを行動ごとの三角形に分け、Q値を表示/着色する設計になっている
env.render_q(agent.Q)
