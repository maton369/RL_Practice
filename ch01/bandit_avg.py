import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent


# このスクリプトは、多腕バンディット + ε-greedy エージェントの学習曲線を
# 「複数試行（runs回）実行して平均を取る」ことで、より安定した評価を行うものです。
#
# 単一試行だけだと、初期の乱数（環境の腕確率や探索の結果）に強く左右されて
# 学習曲線が大きくブレます。
# そこで Monte Carlo 的に runs 回独立に実験し、ステップごとの平均性能を推定します。
#
# 強化学習の実験では一般的に：
# - 乱数による分散が大きい（探索・初期化・環境乱数の影響）
# - 1回の結果は「たまたま良かった/悪かった」が起きる
# ため、「平均曲線」や「分散（誤差帯）」を出すのが定石です。

runs = 200  # 独立試行の回数（ランダム性を平均化して性能を推定するため）
steps = 1000  # 1試行あたりのステップ数（腕を引く回数）
epsilon = 0.1  # ε-greedy の探索率（10%探索、90%活用）

# all_rates[run, step] に「run回目の試行の step までの平均報酬」を保存する2次元配列
# shape は (runs, steps) = (200, 1000)
#
# ここで rates は
#   rates[t] = (t+1 ステップまでの累積報酬) / (t+1)
# なので、学習の進行に伴いどれだけ良い腕に集中できているかが見える指標です。
all_rates = np.zeros((runs, steps))  # (200, 1000)

for run in range(runs):
    # 各 run は独立に環境・エージェントを初期化する。
    # Bandit() は腕ごとの成功確率 p_a をランダム生成するため、
    # run ごとに「別のタスク（別のバンディット問題）」を解く形になっている点に注意。
    #
    # もし「同じバンディット環境で探索の乱数だけ平均化したい」なら、
    # bandit を run の外で固定し、エージェントだけ初期化する設計にする。
    bandit = Bandit()
    agent = Agent(epsilon)

    total_reward = 0  # この run における累積報酬（合計）
    rates = []  # この run における step ごとの平均報酬を保存

    for step in range(steps):
        # ε-greedy による行動選択：
        # - 確率 ε で探索（ランダムな腕）
        # - 確率 1-ε で活用（推定価値 Q が最大の腕）
        action = agent.get_action()

        # 選んだ腕を引いて報酬を観測（ベルヌーイ報酬 0/1）
        reward = bandit.play(action)

        # 観測した (action, reward) で価値推定 Q(action) を更新
        # 更新はサンプル平均（alpha = 1/n）：
        #   Q <- Q + (reward - Q) / n
        agent.update(action, reward)

        # 累積報酬を更新
        total_reward += reward

        # ここで記録しているのは「平均報酬」：
        #   rate_t = total_reward / (t+1)
        # 定常バンディットでは、学習が進むと「高確率の腕を引く割合」が増え、
        # rate_t が上昇していくことが期待される。
        # ただし ε が固定だと探索が残り続けるため、最良腕の期待値より少し下に落ち着く傾向がある。
        rates.append(total_reward / (step + 1))

    # run回目の時系列データを all_rates に格納
    all_rates[run] = rates

# runs 回の曲線を step ごとに平均する。
# avg_rates[t] は「t ステップ目（0-index）時点での平均報酬」を runs 回平均した値。
#
# これは Monte Carlo 推定として：
#   avg_rates[t] ≈ E[ rate_t ]
# を求めていることに対応する。
avg_rates = np.average(all_rates, axis=0)

# --- 可視化 ---
# 平均曲線は「典型的な学習挙動」を表しやすい。
# さらに厳密にやるなら、標準偏差や標準誤差を計算して誤差帯（shaded area）を描くと、
# 学習の不確実性（run間のばらつき）も見えるようになる。
plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.show()
