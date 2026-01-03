# このスクリプトは、GridWorld 環境に対して
# 「価値反復（Value Iteration）」を実装した例です。
#
# 価値反復は、動的計画法（Dynamic Programming; DP）の代表的アルゴリズムであり、
# 最適価値関数 V*(s) をベルマン最適方程式（Bellman optimality equation）の
# 固定点として求めます。
#
# 方策反復（Policy Iteration）との違いは：
# - 方策反復： (評価) V^π を計算 → (改善) π を更新 → … を繰り返す
# - 価値反復： V を直接「最適性（max）」で更新し続け、最後に greedy 方策を作る
#
# 価値反復は「評価と改善を1つの更新に融合したもの」と捉えられます。
# そのため、方策評価を厳密に収束させる必要がなく、実装がシンプルになりやすいです。

# ------------------------------------------------------------
# import パス調整（ファイル実行時のみ）
# ------------------------------------------------------------
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy


def value_iter_onestep(V, env, gamma):
    """
    価値反復の「1 sweep」（全状態に対して1回ベルマン最適更新を適用）を行う。

    入力：
    - V: 現在の価値推定（V_k）
    - env: GridWorld（遷移と報酬を提供）
    - gamma: 割引率 γ

    出力：
    - 更新後の V（V_{k+1}）。この関数内で V を上書き更新して返す。

    理論：
    最適価値関数 V*(s) はベルマン最適方程式を満たす：

        V*(s) = max_a E[ r + γ V*(s') | s,a ]

    決定論的遷移なら期待値の部分は

        E[...] = r(s,a,s') + γ V*(s')

    となるので、更新則（価値反復の基本ステップ）は

        V_{k+1}(s) = max_a [ r(s,a,s') + γ V_k(s') ]

    である。

    この「max」が方策改善に相当し、
    V の更新だけで最適方策に向かう点が価値反復の核心である。
    """
    for state in env.states():
        # ゴールは終端として扱い、価値を 0 と置くのが一般的。
        # （ゴール到達後に報酬が継続しない episodic 環境として）
        if state == env.goal_state:
            V[state] = 0
            continue

        # 各行動の「1ステップ先読み価値」を列挙し、最大を取る
        # action_values[action] = r + γ V(next_state)
        action_values = []

        for action in env.actions():
            # モデルを直接参照して次状態と報酬を得る（モデルベースDP）
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)

            # 1ステップベルマンバックアップ：
            #   value = r + γ V_k(s')
            value = r + gamma * V[next_state]
            action_values.append(value)

        # ベルマン最適バックアップ：
        #   V_{k+1}(s) = max_a value(s,a)
        V[state] = max(action_values)

    return V


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    """
    価値反復（Value Iteration）のメインループ。

    入力：
    - V: 価値関数の初期値（通常は 0 初期化など）
    - env: GridWorld
    - gamma: 割引率 γ
    - threshold: 収束判定の閾値
    - is_render: 途中経過を可視化するかどうか

    出力：
    - 収束した V（最適価値関数 V* の近似）

    収束判定：
        δ = max_s |V_{k+1}(s) - V_k(s)|
      を計算し、δ < threshold で停止する。

    理論的背景：
    γ < 1 のとき、ベルマン最適オペレータ T は収縮写像になるため、
    反復

        V_{k+1} = T V_k

    は一意な固定点 V* に収束することが保証される（標準的な結果）。
    """
    while True:
        # 学習の様子を見たい場合、更新前の V を描画する
        if is_render:
            env.render_v(V)

        # 収束判定のために更新前の V を保存
        old_V = V.copy()

        # 1 sweep 更新（ベルマン最適バックアップを全状態へ適用）
        V = value_iter_onestep(V, env, gamma)

        # 最大更新量 δ を計算（max norm）
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # δ が十分小さければ収束とみなして停止
        if delta < threshold:
            break

    return V


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 実行例
    # ------------------------------------------------------------
    # 価値関数の初期化：未定義状態は 0
    V = defaultdict(lambda: 0)

    # 環境生成
    env = GridWorld()

    # 割引率 γ
    gamma = 0.9

    # 価値反復で V* を求める（途中経過を render_v で表示）
    V = value_iter(V, env, gamma)

    # V* が得られたら、それに対して greedy 方策 π*(s)=argmax_a[r+γV*(s')] を作る
    # （価値反復は価値を求める手法なので、方策は最後に抽出する）
    pi = greedy_policy(V, env, gamma)

    # 最終的な価値と方策を可視化
    env.render_v(V, pi)
