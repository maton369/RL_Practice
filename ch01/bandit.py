import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    多腕バンディット（Multi-Armed Bandit; MAB）の環境クラス。

    この環境は「状態」を持たない（または状態が常に一定）ので、
    強化学習の中でも最も単純な「探索と活用（exploration vs exploitation）」問題になる。

    - 腕（arm） i を引くと、確率 r_i で報酬 1、確率 (1 - r_i) で報酬 0 を得る。
    - 各腕の真の成功確率 r_i は環境の内部パラメータで、エージェントは知らない。
    - エージェントの目的は、試行回数が増えたときに累積報酬を最大化すること。

    ここでは報酬分布はベルヌーイ分布：
        reward ~ Bernoulli(r_i)
    となっている（成功なら1、失敗なら0）。
    """

    def __init__(self, arms=10):
        # 各腕の成功確率 r_i を [0,1) の一様乱数で初期化
        # rates[i] が腕 i の「真の平均報酬（期待報酬）」に対応する。
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """
        指定した腕 arm を 1 回引いて、報酬を返す。

        - rate = r_arm を取り出す
        - 一様乱数 u ~ Uniform(0,1) を生成し、u < rate なら成功（報酬1）
          そうでなければ失敗（報酬0）

        これは「成功確率 rate のベルヌーイ試行」を手作業で実装している形。
        """
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    """
    ε-greedy 方策で行動選択するバンディット用エージェント。

    エージェントは各腕 a について「その腕の価値推定値」Q(a) を持つ。
    ここでの Q(a) は RL の表記に合わせると「行動価値（action-value）」だが、
    バンディットなので状態がなく Q(s,a) ではなく Q(a) で十分。

    学習目標：
        各腕 a の期待報酬 E[r | a] を推定しつつ、
        高い期待報酬を持つ腕をより頻繁に選ぶことで累積報酬を増やす。

    行動選択（探索と活用）：
        - 確率 ε で探索：ランダムに腕を選ぶ（未知の腕を試す）
        - 確率 1-ε で活用：現時点で Q が最大の腕を選ぶ（稼げそうな腕を引く）

    価値更新：
        ここでは「サンプル平均（sample-average）」で Q を更新する。
        腕 a を n 回引いたときの推定値 Q_n(a) は逐次更新で

            Q_n(a) = Q_{n-1}(a) + (r_n - Q_{n-1}(a)) / n

        となる。これは「誤差駆動更新」

            estimate <- estimate + alpha * (target - estimate)

        の特殊ケースで、学習率が alpha = 1/n に自動で減衰する形。
        定常環境（真の r_i が時間で変わらない）では自然な推定法である。
    """

    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon

        # Qs[a]：腕 a の価値推定 Q(a)
        # 初期値 0 は「楽観的初期値」ではないので、初期は探索に依存しやすい。
        self.Qs = np.zeros(action_size)

        # ns[a]：腕 a を引いた回数 n(a)
        # サンプル平均更新で分母に必要。
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        """
        行動 action を選んで報酬 reward を観測した後に、Q(action) を更新する。

        サンプル平均の逐次更新：
            n <- n + 1
            Q <- Q + (reward - Q) / n

        - (reward - Q) は現在推定と観測の「誤差（prediction error）」。
        - 1/n は学習率（ステップサイズ）で、サンプルが増えるほど更新幅が小さくなる。
        - この更新を繰り返すと、Q(a) はその腕の平均報酬推定値に一致していく。
        """
        # この腕を引いた回数を 1 増やす
        self.ns[action] += 1

        # 逐次的に標本平均へ更新（数式上、これで厳密に sample average と一致する）
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """
        ε-greedy に従って次の腕を選ぶ。

        - 一様乱数 u を生成し、u < ε なら探索（ランダム選択）
        - そうでなければ活用（argmax Q）

        注意：
        - ε が小さすぎると初期に誤った腕へ固着しやすい（探索不足）
        - ε が大きすぎると良い腕を見つけても十分活用できない（活用不足）
        - バンディットはこのトレードオフの典型例
        """
        if np.random.rand() < self.epsilon:
            # 探索：一様にランダムな腕を選ぶ
            return np.random.randint(0, len(self.Qs))
        # 活用：現在の推定で最良の腕を選ぶ（複数同率なら最初の index になる点に注意）
        return np.argmax(self.Qs)


if __name__ == "__main__":
    # 試行回数（ステップ数）：腕を引く回数
    steps = 1000

    # ε-greedy の探索率
    # 例：epsilon=0.1 なら 10% 探索、90% 活用
    epsilon = 0.1

    # 環境とエージェントを生成
    bandit = Bandit()
    agent = Agent(epsilon)

    # 累積報酬（return ではなく、単純な「合計」）
    total_reward = 0

    # 可視化用ログ：
    # total_rewards[t]：t ステップ目までの累積報酬
    # rates[t]：t ステップ目までの平均報酬（累積 / ステップ数）
    total_rewards = []
    rates = []

    for step in range(steps):
        # 行動選択（探索 or 活用）
        action = agent.get_action()

        # 環境から報酬を得る（ベルヌーイ報酬）
        reward = bandit.play(action)

        # 観測した (action, reward) で価値推定を更新
        agent.update(action, reward)

        # 統計を更新
        total_reward += reward
        total_rewards.append(total_reward)

        # 平均報酬（学習が進むほど最良腕の成功確率に近づくのが理想）
        rates.append(total_reward / (step + 1))

    # 最終的に得られた累積報酬を出力
    print(total_reward)

    # --- 可視化 1：累積報酬 ---
    # 良い腕を見つけて活用できるようになると、傾きが大きくなりやすい。
    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    # --- 可視化 2：平均報酬 ---
    # 理想的には「最良腕の成功確率（max rate）」付近へ収束する。
    # ただし ε-greedy は探索でたまに悪い腕も引くため、
    # 収束先は max rate より少し低くなることが多い（ε が固定の場合）。
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()
