# 目的：
# - NumPy 2.0 + 旧Gymの互換性問題（np.bool8）が原因で落ちているので、
#   Gymnasium（後継）へ移行して CartPole をランダム方策で動かす。
#
# 背景（あなたのエラーの原因）：
# - 旧Gymは「NumPy 2.0をサポートしない」ことが明示されており、
#   内部で np.bool8 を参照する箇所があり NumPy 2.0 で AttributeError になる。
# - Gymnasiumは保守されているドロップイン置換で、APIも新しい形（terminated/truncated）になっている。
#
# まずは環境の入れ替え（推奨）：
#   pip uninstall -y gym
#   pip install -U "gymnasium[classic-control]"
#
# 補足：
# - classic-control の描画には pygame が必要なことが多いが、
#   gymnasium[classic-control] で一緒に入る構成が一般的。
# - CartPole-v0 は古く、CartPole-v1 を使うのが推奨。

import numpy as np
import gymnasium as gym  # 旧: import gym


# ------------------------------------------------------------
# 環境の生成
# ------------------------------------------------------------
# Gymnasiumでは render_mode を make() 時に指定する。
# - "human": ウィンドウ表示（ローカルで観察したいとき）
# - "rgb_array": 画像配列が欲しいとき（学習ログや動画保存向け）
env = gym.make("CartPole-v1", render_mode="human")

# ------------------------------------------------------------
# エピソード開始：reset()
# ------------------------------------------------------------
# Gymnasiumの reset() は (observation, info) を返す。
# - observation: 状態（観測）ベクトル
# - info: 追加情報（デバッグ用）
#
# seed を渡すと再現性が上がる（任意）。
obs, info = env.reset(seed=0)

# ------------------------------------------------------------
# メインループ：step()
# ------------------------------------------------------------
# Gymnasiumの step() は5つ返す：
#   (observation, reward, terminated, truncated, info)
#
# - terminated: タスクの失敗/成功など「MDPとしての終端」
# - truncated: 時間制限など「外的要因による打ち切り」
#
# 旧Gymの done は terminated OR truncated に相当する。
terminated = False
truncated = False

try:
    while not (terminated or truncated):
        # ランダム行動（最も単純な方策）
        # np.random.choice([0,1]) でも良いが、action_space.sample() が汎用的。
        action = env.action_space.sample()

        # 1ステップ進める
        obs, reward, terminated, truncated, info = env.step(action)

        # ここでは学習しないので、観測 obs を保持して次ループへ進むだけ。
        # 強化学習をするなら、このタイミングで遷移
        #   (s, a, r, s', terminated/truncated)
        # を使って価値関数や方策を更新する。

finally:
    # ウィンドウ等のリソース解放
    env.close()
