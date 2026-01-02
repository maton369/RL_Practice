import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent


class NonStatBandit:
    """
    非定常（Non-Stationary）な多腕バンディット環境。

    通常のバンディット（定常バンディット）では、
    各腕 a の成功確率 p_a は時間によらず固定である。
    しかし現実の推薦・広告・ユーザ反応などは時間で変わりうるため、
    真の期待報酬 p_a(t) が変化する「非定常環境」を考える必要がある。

    このクラスは、その非定常性を「各ステップで腕ごとの成功確率 rates をランダムウォークさせる」
    ことで実現している。

    - 初期成功確率 rates[a] を [0,1) からランダムに生成
    - play() が呼ばれるたびに全腕の rates にノイズを加えて変化させる（ドリフトさせる）
      rates <- rates + 0.1 * N(0,1)

    注意：
    - rates はノイズで 0 未満や 1 超になりうる。
      その場合「確率」としての意味が壊れるので、
      厳密には clip（例：np.clip(rates, 0, 1)）するのが自然。
      ただし本コードは「非定常で推定が難しくなる」雰囲気を掴む教材として動く。
    """

    def __init__(self, arms=10):
        self.arms = arms
        # 各腕の「その時点での成功確率」をランダム初期化
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """
        腕 arm を 1 回引き、報酬 0/1 を返す。

        手順：
        1. 現時点の成功確率 rate = rates[arm] を取得
        2. 環境を時間発展させる：全腕の rates にノイズを加える（非定常化）
        3. rate に従うベルヌーイ試行で報酬を返す

        ポイント：
        - 報酬を生成するのに使う rate は「更新前」の rates[arm] を使っている。
          つまり「腕を引いた直後に環境が変化する」モデルになっている。
        """
        rate = self.rates[arm]

        # 各ステップで全腕にガウスノイズを足して、成功確率が時間変化するようにする
        # 0.1 はドリフトの大きさ（変動の激しさ）を決めるハイパーパラメータ
        self.rates += 0.1 * np.random.randn(self.arms)  # Add noise

        # ベルヌーイ報酬：rate の確率で 1、そうでなければ 0
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    """
    固定学習率（constant step-size）で価値推定を行う ε-greedy エージェント。

    非定常環境では「過去のデータは古くなる」ため、
    定常環境で自然なサンプル平均（学習率 1/n）が弱点になる。

    理由：
    - サンプル平均は n が増えると学習率 1/n が 0 に近づき、
      新しい観測をほとんど反映できなくなる（追従性が落ちる）。
    - 非定常では「最近の情報を重視する」必要がある。

    そこで固定学習率 alpha を使って指数移動平均（Exponential Recency-Weighted Average）にする：
        Q <- Q + alpha * (reward - Q)

    これは RL の一般的な更新：
        estimate <- estimate + alpha * (target - estimate)
    と同型で、alpha が「どれくらい最近の情報を重視するか」を決める。

    alpha の解釈：
    - alpha が大きい：新しい観測を強く反映 → 追従性は高いが分散が大きい（ブレる）
    - alpha が小さい：推定が滑らか → 分散は小さいが変化に追従しにくい

    このクラスは「非定常バンディットには固定 alpha が有利になりやすい」という教科書的比較を示す。
    """

    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        # Qs[a]：腕 a の価値推定（状態なしなので Q(a) のみで十分）
        self.Qs = np.zeros(actions)
        # 固定学習率 alpha（0 < alpha <= 1 を想定）
        self.alpha = alpha

    def update(self, action, reward):
        """
        固定学習率による誤差駆動更新：
            Q(a) <- Q(a) + alpha * (reward - Q(a))

        サンプル平均と違い、過去のデータを指数的に忘れていくため、
        非定常な真値 p_a(t) の変化に追従しやすい。
        """
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        """
        ε-greedy により行動選択。

        - 確率 ε：探索（ランダム選択）
        - 確率 1-ε：活用（argmax Q）

        非定常環境では「環境が変わった結果、最良腕が入れ替わる」ことがあるため、
        探索が常に少し残る設計（固定 ε）は理にかなっている場合が多い。
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


# --- 実験設定 ---
runs = 200  # 独立試行回数（分散を平均化して性能比較する）
steps = 1000  # 1試行あたりのステップ数
epsilon = 0.1  # ε-greedy の探索率
alpha = 0.8  # 固定学習率（大きめ：追従性重視。ただしブレも大きくなりやすい）

# 比較するエージェントの種類：
# 1) sample average（サンプル平均更新：定常向き）
# 2) alpha const update（固定学習率更新：非定常向き）
agent_types = ["sample average", "alpha const update"]
results = {}  # 各手法の「平均学習曲線」を保存する辞書

for agent_type in agent_types:
    # all_rates[run, step] に、run 回目の step 時点の平均報酬（累積/step）を保存
    all_rates = np.zeros((runs, steps))  # (200, 1000)

    for run in range(runs):
        # エージェントを切り替え
        # - Agent は bandit.py 由来で、サンプル平均（1/n）で Q を更新するタイプ
        # - AlphaAgent は本ファイルで定義した固定 alpha 更新タイプ
        if agent_type == "sample average":
            agent = Agent(epsilon)
        else:
            agent = AlphaAgent(epsilon, alpha)

        # 非定常バンディット環境を初期化
        # run ごとに rates の初期状態も変わるので、別タスクを平均している形になる。
        bandit = NonStatBandit()

        total_reward = 0
        rates = []

        for step in range(steps):
            # 行動選択（探索 or 活用）
            action = agent.get_action()

            # 環境から報酬を観測（この後、環境パラメータ rates が変化する）
            reward = bandit.play(action)

            # 観測を使って価値推定を更新
            agent.update(action, reward)

            # 累積報酬と平均報酬を更新
            total_reward += reward
            rates.append(total_reward / (step + 1))

        # run の時系列を保存
        all_rates[run] = rates

    # step ごとに runs 回の平均を取ることで、典型的挙動を推定
    avg_rates = np.average(all_rates, axis=0)
    results[agent_type] = avg_rates

# --- plot ---
# 非定常環境では一般に：
# - サンプル平均（1/n）は後半で更新が弱くなり、変化に追従しにくくなる
# - 固定 alpha は追従できるため、長期的に平均報酬が上回ることが多い
#
# ただし alpha が大きすぎるとノイズに振り回されて不安定にもなるため、
# alpha は環境の変化スピード（ここでは 0.1 のノイズ強度）に合わせて調整が必要。
plt.figure()
plt.ylabel("Average Rates")
plt.xlabel("Steps")
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()
