import numpy as np
import matplotlib.pyplot as plt


def argmax(xs):
    """
    xs（数値列）の最大値を取るインデックスを返す argmax の自作版。

    なぜ自作するのか：
    - np.argmax は最大値が複数あるとき「最初の要素」を返す（決め打ちタイブレーク）
    - 強化学習では、初期の Q 値が同じ（例：全部0）になりやすく、
      そのとき np.argmax を使うと「常に同じ行動（例：action=0）ばかり選ぶ」偏りが起きる。
    - この偏りは探索・学習の進み方や最終方策に影響することがある。
      （特に deterministic な環境や、同率最適行動がある環境では顕著）

    そこでこの関数では：
    - 最大値を取る候補インデックスをすべて列挙し、
    - 複数あればランダムに選ぶ
    という「ランダムタイブレーク（random tie-breaking）」を実装している。

    入力：
    - xs: list/ndarray 等の数値列

    出力：
    - 最大値を取るインデックス（int）

    注意：
    - len(idxes)==0 の分岐は通常は起きない（max(xs) が必ず存在するため）。
      ただし xs が空配列だった場合は max(xs) 自体が例外になるので、
      厳密には「xs が空でない」前提で使うべき関数である。
    """
    # xs の中で最大値 max(xs) と等しい要素のインデックスを全部集める
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]

    # 最大値候補が1つだけなら、そのインデックスを返す
    if len(idxes) == 1:
        return idxes[0]

    # 通常は起きないが、念のため候補が0なら全体からランダムに選ぶ
    # （ただし xs が空なら np.random.choice(len(xs)) はエラーになる点に注意）
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    # 最大値候補が複数なら、その中からランダムに1つ選ぶ（タイブレーク）
    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q, state, epsilon=0, action_size=4):
    """
    Q(s,a) から ε-greedy 方策 π(·|s) を生成して「行動確率分布」を返す。

    目的：
    - 方策改善（policy improvement）の最も基本形として
      Q の大きい行動を優先しつつ、探索も残す分布を作る。

    入力：
    - Q: dict 形式の行動価値関数。キーは (state, action)、値は Q(s,a)
    - state: 方策を計算したい状態 s
    - epsilon: 探索率 ε（0 <= ε <= 1）
    - action_size: 行動数 |A|

    出力：
    - action_probs: dict {action: prob} で表した確率分布 π(a|s)

    理論（ε-greedy）：
    greedy 行動 a* = argmax_a Q(s,a) を基本として、
    - 確率 1-ε で a* を選ぶ（活用）
    - 確率 ε でランダム行動（探索）
    を混ぜる。

    実装では「まず全行動に ε/|A| を配り、greedy 行動に (1-ε) を上乗せ」しているので

        π(a|s) =
          1-ε + ε/|A|   (a = a*)
          ε/|A|         (otherwise)

    になる。

    重要ポイント（タイブレーク）：
    - greedy 行動の決定に argmax(qs) を使っており、
      同率最大のときはランダムに選ぶため、方策が偏りにくい。
      これは初期学習段階（Qが同値）で特に有効。
    """
    # 状態 state における各行動の Q(s,a) を配列として取り出す
    qs = [Q[(state, action)] for action in range(action_size)]

    # 最大Qを取る行動（同率ならランダム）を選ぶ
    max_action = argmax(qs)  # OR np.argmax(qs)

    # 探索成分：全行動に ε/|A| を割り当てる
    base_prob = epsilon / action_size
    action_probs = {
        action: base_prob for action in range(action_size)
    }  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}

    # 活用成分：(1-ε) を greedy 行動へ追加
    action_probs[max_action] += 1 - epsilon

    return action_probs


def plot_total_reward(reward_history):
    """
    エピソードごとの「総報酬（Total Reward）」の推移をプロットする可視化関数。

    背景（RL実験での意味）：
    - 強化学習では「学習が進んだか」を確認するために、
      エピソードごとの報酬（累積報酬や平均報酬）をログして可視化することが多い。
    - 特に episodic なタスク（GridWorld等）では、
      1エピソードの中で得た報酬の合計

          R_total = Σ_t R_{t+1}

      をエピソード index に対して描くと、
      方策が改善してゴール到達が増えたか、罰則を回避できたか、などの傾向が見える。

    入力：
    - reward_history: list[float] / list[int]
        reward_history[e] が e 番目のエピソードで得た総報酬

    出力：
    - matplotlib の折れ線グラフを表示

    注意：
    - 単発の学習曲線は分散が大きいので、複数 run の平均や移動平均を併記するとより見やすい。
    """
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # x軸はエピソード番号、y軸は総報酬
    plt.plot(range(len(reward_history)), reward_history)

    plt.show()
