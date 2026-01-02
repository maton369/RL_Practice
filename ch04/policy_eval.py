# このスクリプトは、GridWorld 環境に対して
# 「方策評価（Policy Evaluation）」を動的計画法（DP）として実装した例です。
#
# ここでやっていること：
# - 方策 π が固定されているとき、その方策の下での状態価値関数 V^π(s) を求める。
# - 価値関数はベルマン期待方程式（Bellman expectation equation）の固定点として定義される。
# - それを反復更新（Iterative Policy Evaluation）で数値的に収束させる。
#
# 重要なRLの構造：
# - 「環境が完全に分かっている（モデルあり）」前提のDP手法である。
#   つまり次状態 next_state や報酬 reward を環境関数として直接呼べる。
# - モデルフリー（TDやMC）のようにサンプルから推定するのではなく、
#   期待値（全行動の和）を毎回きっちり計算する。

# ------------------------------------------------------------
# スクリプト実行時の import パス調整（教材プロジェクトでよくあるやつ）
# ------------------------------------------------------------
# __file__ が存在する（= ファイルとして実行されている）場合に限り、
# 1つ上のディレクトリを sys.path に追加して common/ を import できるようにする。
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    """
    ベルマン期待更新を「1回だけ」全状態に対して適用する（1 sweep）。

    入力：
    - pi: 方策 π。pi[state] = {action: prob} という確率分布を持つ辞書
    - V:  価値関数の推定値。V[state] = value
    - env: GridWorld 環境（遷移関数 next_state と報酬関数 reward を提供）
    - gamma: 割引率 γ

    出力：
    - 更新後の V（この関数内で V を上書き更新して返す）

    理論：
    固定方策 π の下での状態価値は

        V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ r(s,a,s') + γ V^π(s') ]

    で定義される（ベルマン期待方程式）。

    この GridWorld は遷移が決定論的なので、P の和は実質的に「次状態は1つ」になる。
    よって実装は

        V(s) <- Σ_a π(a|s) [ r + γ V(s') ]

    という形の更新になる。
    """
    for state in env.states():
        # 終端（ゴール）状態は、その先の価値を 0 と置くのが一般的。
        # （ゴール到達後にエピソードが終了し、以後の報酬が無いという扱い）
        if state == env.goal_state:
            V[state] = 0
            continue

        # 状態 state における行動分布 π(a|s) を取り出す
        action_probs = pi[state]

        # new_V は V_{k+1}(state) の一時変数（期待値を畳み込んでいく）
        new_V = 0

        # 全行動 a に対して期待値を計算（Σ_a π(a|s) * [ ... ]）
        for action, action_prob in action_probs.items():
            # 次状態 s' を取得（この環境では決定論的遷移）
            next_state = env.next_state(state, action)

            # 報酬 r = R(s,a,s') を取得（この環境は「到達先セルの報酬」）
            r = env.reward(state, action, next_state)

            # ベルマン期待更新：
            #   new_V += π(a|s) * ( r + γ V(s') )
            new_V += action_prob * (r + gamma * V[next_state])

        # 同期更新っぽく見えるが、実装上は V[state] をその場で書き換えるため、
        # 更新順序によっては「非同期（Gauss-Seidel）成分」が混ざる。
        # 小規模環境ではどちらでも収束しやすいが、理論と一致させたい場合は
        # old_V を参照して new_V を別辞書に書く設計もある。
        V[state] = new_V

    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    """
    反復方策評価（Iterative Policy Evaluation）：
    eval_onestep を繰り返して V を固定点（V^π）へ収束させる。

    入力：
    - pi: 方策 π
    - V:  価値関数の初期推定（通常は 0 初期化など）
    - env: 環境
    - gamma: 割引率 γ
    - threshold: 収束判定の閾値（最大ノルム差分がこれ未満になったら停止）

    出力：
    - 収束した V（V^π の近似）

    収束判定：
        δ = max_s |V_new(s) - V_old(s)|
      を計算し、δ < threshold で終了する。

    理論的背景：
    γ < 1 のとき、ベルマン期待オペレータは収縮写像になるため、
    反復により一意な固定点 V^π へ収束することが期待される。
    """
    while True:
        # 収束判定のため、更新前の V を保存（同じ state キー集合で比較するため copy する）
        old_V = V.copy()

        # 1 sweep 更新（ベルマン期待更新を全状態に適用）
        V = eval_onestep(pi, V, env, gamma)

        # 最大更新量 δ = max_s |V(s) - old_V(s)| を求める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # δ が十分小さければ収束とみなす
        if delta < threshold:
            break

    return V


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 実験のセットアップ
    # ------------------------------------------------------------
    env = GridWorld()
    gamma = 0.9

    # 方策 π の初期化：
    # defaultdict を使い、どの state を参照しても
    # 「4行動が一様（0.25ずつ）」の方策を返すようにしている。
    #
    # これはランダム方策（uniform random policy）であり、
    # 方策評価の教材として最も分かりやすいベースライン。
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    # 価値関数 V の初期化：
    # defaultdict により未定義 state の値は 0 になる。
    # 反復DPは初期値に依らず収束することが多いが、
    # 初期値によって収束速度や途中の値の見え方は変わる。
    V = defaultdict(lambda: 0)

    # ------------------------------------------------------------
    # 方策評価を実行
    # ------------------------------------------------------------
    V = policy_eval(pi, V, env, gamma)

    # ------------------------------------------------------------
    # 可視化：V(s) と π(s) を同時に描画
    # ------------------------------------------------------------
    # - セル内に V(s) の数値（またはヒートマップ）
    # - 方策の矢印（この場合は一様なので、最大確率が同率になり複数矢印になる可能性がある）
    env.render_v(V, pi)
