from collections import deque
import random
import numpy as np
import gymnasium as gym


# ------------------------------------------------------------
# ReplayBuffer（経験再生バッファ）
# ------------------------------------------------------------
# DQN系アルゴリズムでほぼ必須になる部品。
# 目的は「オンラインで得た遷移データをためておき、あとからランダムに取り出して学習する」こと。
#
# なぜ必要か（アルゴリズム的理由）：
# - 連続した遷移 (s_t, a_t, r_{t+1}, s_{t+1}) は強い相関を持つ。
#   そのまま逐次学習すると、勾配更新が偏りやすく学習が不安定になりやすい。
# - ReplayBuffer に貯めた遷移をランダムサンプルすることで、
#   データが（完全ではないが）i.i.d. に近づき、学習が安定しやすくなる。
# - 1回観測した遷移を何度も学習に再利用でき、サンプル効率も改善する。
#
# ここでは「最小の一様ランダムサンプリングのReplayBuffer」を実装している。
# （Prioritized Replay 等はさらに拡張版）
# ------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        # deque(maxlen=buffer_size) は容量上限を超えると古い要素から自動で捨てる
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        # 遷移（transition）を1件追加する。
        # 強化学習の標準形は：
        #   (s_t, a_t, r_{t+1}, s_{t+1}, done)
        # done は「この遷移でエピソードが終わったか」を表すフラグ。
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        # バッファに現在入っている遷移件数
        return len(self.buffer)

    def get_batch(self):
        # バッファから batch_size 件を「重複なし」でランダム抽出する。
        # ※ len(buffer) < batch_size のときは例外になるので、
        #    実戦では「溜まるまで学習しない」等のガードを入れるのが普通。
        data = random.sample(self.buffer, self.batch_size)

        # ミニバッチ学習しやすい形に整形する。
        #
        # state / next_state は観測ベクトルなので stack して (B, obs_dim) にする。
        # action / reward / done は (B,) のベクトルにする。
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])

        # done はブートストラップ停止に使うので、0/1 にしておくと便利：
        #   target = r + gamma * (1-done) * max_a Q(s',a)
        done = np.array([x[4] for x in data]).astype(np.int32)

        return state, action, reward, next_state, done


# ------------------------------------------------------------
# Gym → Gymnasium への修正点
# ------------------------------------------------------------
# 旧GymはNumPy 2.0で壊れる（np.bool8）問題が出ることがあるため、
# Gymnasium（後継）を使うのが推奨。
#
# API変更（重要）：
# - env.reset() -> (obs, info)
# - env.step(a) -> (obs, reward, terminated, truncated, info)
#   done は terminated or truncated に相当する。
# ------------------------------------------------------------
env = gym.make("CartPole-v1")  # v0は古いのでv1推奨
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

# ------------------------------------------------------------
# データ収集（学習はせず、遷移をバッファに溜めるだけ）
# ------------------------------------------------------------
for episode in range(10):
    # reset() は (obs, info) を返す
    state, info = env.reset(seed=episode)  # seedを変えるとエピソードが少し多様になる

    terminated = False
    truncated = False

    while not (terminated or truncated):
        # ここでは固定行動 action=0（左に押す）にしている。
        # データを多様にしたいなら env.action_space.sample() や ε-greedy を使う。
        action = 0

        # step() は 5要素を返す
        next_state, reward, terminated, truncated, info = env.step(action)

        # done は旧Gym互換の意味でまとめて扱う
        done = bool(terminated or truncated)

        # 遷移をReplayBufferに保存
        replay_buffer.add(state, action, reward, next_state, done)

        # 次ステップへ
        state = next_state

env.close()

# ------------------------------------------------------------
# バッファからミニバッチを取り出して形状確認
# ------------------------------------------------------------
# 注意：10エピソード・固定行動でもCartPoleはそこそこ長く続くことが多いが、
#       万一データが32未満なら get_batch() が例外になる。
#       実戦では len(replay_buffer) >= batch_size のチェックを入れる。
state, action, reward, next_state, done = replay_buffer.get_batch()

print(state.shape)  # (32, 4)
print(action.shape)  # (32,)
print(reward.shape)  # (32,)
print(next_state.shape)  # (32, 4)
print(done.shape)  # (32,)
