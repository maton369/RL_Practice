# このスクリプトは、GridWorld 環境に対して
# 「方策反復（Policy Iteration）」を実装した例です。
#
# 方策反復は、動的計画法（DP）の代表的アルゴリズムであり、
# 以下の2ステップを交互に繰り返して最適方策 π* を求めます。
#
# 1) 方策評価（Policy Evaluation）
#    固定方策 π の下での価値関数 V^π をベルマン期待方程式の固定点として求める。
#
# 2) 方策改善（Policy Improvement）
#    得られた V^π を使って、各状態で「価値を最大にする行動」を選ぶ greedy 方策へ更新する。
#
# これを繰り返すと、（有限状態・有限行動・γ<1 の設定では）方策は単調に改善され、
# 最終的に最適方策 π* に到達することが知られています。
#
# 直感：
# - 評価で「今の方策は各状態でどれくらい得か」を数値化し、
# - 改善で「その数値が最大になる行動を選ぶように方策を作り直す」
# という流れです。

# ------------------------------------------------------------
# import パス調整（ファイル実行時のみ）
# ------------------------------------------------------------
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_eval import policy_eval


def argmax(d):
    """
    dict d の value が最大になる key を返すヘルパー関数。

    注意：
    - 最大値が複数ある場合、この実装は「最後に見つかった最大キー」を返す。
      （for ループで上書きしていくため）
    - そのため、同率最大があるとタイブレークが実装依存になる。
      方策反復の最終結果が変わる可能性がある（ただし最適方策が複数存在する場合のみ）。
    - タイブレークを明示したい場合は「最初に見つかった最大キー」や
      「ランダムに選ぶ」などを実装するのがよい。
    """
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    """
    価値関数 V をもとに greedy 方策 π' を構成する（方策改善ステップ）。

    入力：
    - V: 状態価値関数 V(s)（通常は policy_eval で求めた V^π）
    - env: GridWorld（遷移と報酬を提供）
    - gamma: 割引率 γ

    出力：
    - pi: greedy 方策（決定的方策）
         pi[state] = {action: prob} の形で、最大の行動だけ確率1、他は0。

    理論：
    方策改善定理（Policy Improvement Theorem）より、
    現在の方策 π に対する価値 V^π が分かっているとき、次の greedy 更新

        π'(s) = argmax_a Q^π(s,a)

    を行えば、改善された方策 π' は

        V^{π'}(s) >= V^π(s)  （全状態で）

    を満たす（同等か改善）ことが保証される。

    ここで Q^π(s,a) は

        Q^π(s,a) = E[ r + γ V^π(s') | s,a ]

    であり、遷移が決定論的なら

        Q^π(s,a) = r(s,a,s') + γ V^π(s')

    をそのまま計算すればよい。
    """
    pi = {}

    for state in env.states():
        # action_values[action] = 推定 Q^π(s,a) を格納する辞書
        action_values = {}

        for action in env.actions():
            # 環境モデルを使って次状態と報酬を決定（モデルありDP）
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)

            # 1ステップ先読みの Q 推定（ベルマンの形）
            # Q^π(s,a) = r + γ V^π(s')
            value = r + gamma * V[next_state]
            action_values[action] = value

        # Q が最大の行動（greedy action）を選ぶ
        max_action = argmax(action_values)

        # 方策を {action: prob} の形で保持（可視化 renderer と整合）
        # 決定的方策なので max_action の確率を 1.0 にする
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi


def policy_iter(env, gamma, threshold=0.001, is_render=True):
    """
    方策反復（Policy Iteration）のメインループ。

    初期方策：
    - ここでは一様ランダム方策（各行動 0.25）から開始。

    ループ：
    1) 方策評価：V <- V^π を近似的に計算（policy_eval）
    2) 方策改善：π' <- greedy_policy(V)
    3) π' が π と同じなら収束（最適方策に到達したとみなす）
       そうでなければ π <- π' として続行

    入力：
    - env: GridWorld
    - gamma: 割引率 γ
    - threshold: 方策評価（policy_eval）の収束閾値
    - is_render: 各反復で V と π を可視化するかどうか

    出力：
    - 最終的な方策 π（最適方策 π* のはず）

    理論的ポイント：
    - 有限MDP + γ<1 では、方策反復は有限回で収束する（同じ方策が再登場した時点で停止）。
    - 各改善ステップで価値が単調に上がる（または不変）ので、無限ループしにくい構造になっている。
    """
    # 初期方策：一様ランダム
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    # 価値関数の初期化：未定義 state は 0
    V = defaultdict(lambda: 0)

    while True:
        # --------------------------------------------------------
        # 1) 方策評価（Policy Evaluation）
        # --------------------------------------------------------
        # 現在方策 π の価値 V^π を反復更新で求める。
        # threshold は評価の精度（小さいほど正確だが反復回数が増える）。
        V = policy_eval(pi, V, env, gamma, threshold)

        # --------------------------------------------------------
        # 2) 方策改善（Policy Improvement）
        # --------------------------------------------------------
        # 得られた V^π を使って greedy 方策 π' を作る。
        new_pi = greedy_policy(V, env, gamma)

        # 学習過程を可視化：各反復で価値と方策がどう変わるかを見る
        if is_render:
            env.render_v(V, pi)

        # --------------------------------------------------------
        # 3) 収束判定（方策が変わらなくなったら終了）
        # --------------------------------------------------------
        # new_pi == pi なら、改善しても変わらない＝すでに greedy であり、
        # 方策改善定理の観点から最適方策に到達している（少なくとも局所改善余地がない）。
        #
        # 注意：
        # - pi は最初 defaultdict だが new_pi は通常 dict になる。
        #   Python の等価比較は型が違っても中身が同じなら True になる場合が多いが、
        #   デフォルトファクトリの扱いなどで予期せぬ差異が出る可能性もある。
        #   収束判定を堅牢にするなら「全状態で action_probs が一致するか」を明示的に比較するのも手。
        if new_pi == pi:
            break

        # 方策を更新して次の反復へ
        pi = new_pi

    return pi


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 実行例
    # ------------------------------------------------------------
    env = GridWorld()
    gamma = 0.9

    # 方策反復で最適方策を求める（途中経過は render_v で可視化される）
    pi = policy_iter(env, gamma)
