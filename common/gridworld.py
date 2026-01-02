import numpy as np
import common.gridworld_render as render_helper


class GridWorld:
    """
    典型的な「格子状のMDP（GridWorld）」環境クラス。

    強化学習（RL）では環境は以下を提供するのが基本：
    - 状態集合 S（ここではグリッド上の座標 (y, x)）
    - 行動集合 A（ここでは 4方向：上・下・左・右）
    - 遷移関数 P(s'|s,a)（ここでは決定論的：next_state で一意に決まる）
    - 報酬関数 R(s,a,s')（ここでは「遷移先セルに埋め込まれた報酬」）
    - 終端状態（goal_state など）

    この環境は教材向けにシンプルで、次の特徴を持つ：
    - 壁（wall_state）には侵入できない（ぶつかるとその場に留まる）
    - 盤面外への移動もできない（盤面外へ出ようとするとその場に留まる）
    - ゴール（goal_state）に到達すると done=True になる（ただしその後の遷移はここでは未定義）
    - 報酬は「次状態のセル」で決まる（reward_map[next_state]）

    この手の GridWorld は、動的計画法（価値反復・方策反復）や
    TD学習（SARSA, Q-learning）の最初の題材としてよく使われる。
    """

    def __init__(self):
        # ------------------------------------------------------------
        # 行動の定義
        # ------------------------------------------------------------
        # action_space は取りうる行動の一覧。
        # 0..3 を方向に割り当てる（UP/DOWN/LEFT/RIGHT）。
        # RLアルゴリズムでは、行動を整数IDとして扱うほうが配列で実装しやすい。
        self.action_space = [0, 1, 2, 3]

        # デバッグや可視化用：行動ID -> 意味（文字列）
        self.action_meaning = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        # ------------------------------------------------------------
        # 報酬マップ（環境の地形）
        # ------------------------------------------------------------
        # reward_map は各セルの「到達したときの報酬」を表す。
        # shape は (height, width) = (3,4)。
        #
        # ここで None が置かれているセルは「壁」を表現している（通行不可）。
        # 実際の reward 計算では next_state が wall_state にならないようにガードするので、
        # None が reward として返ることは基本的にない設計。
        #
        # (0,3) が +1 のゴール、(1,3) が -1 の罠のような終端報酬、と読める。
        self.reward_map = np.array([[0, 0, 0, 1.0], [0, None, 0, -1.0], [0, 0, 0, 0]])

        # ------------------------------------------------------------
        # 特殊状態（ゴール / 壁 / スタート）
        # ------------------------------------------------------------
        # 状態は (y, x) のタプルで表す（行→y, 列→x）。
        self.goal_state = (0, 3)  # 到達で終了（done=True）
        self.wall_state = (1, 1)  # 侵入不可の壁
        self.start_state = (2, 0)  # エピソード開始位置

        # エージェントの現在位置（内部状態として保持）
        self.agent_state = self.start_state

    # ------------------------------------------------------------
    # 盤面サイズ関連（便利プロパティ）
    # ------------------------------------------------------------
    @property
    def height(self):
        # グリッドの縦サイズ（行数）
        return len(self.reward_map)

    @property
    def width(self):
        # グリッドの横サイズ（列数）
        return len(self.reward_map[0])

    @property
    def shape(self):
        # numpy 配列としての shape (height, width)
        return self.reward_map.shape

    # ------------------------------------------------------------
    # MDPの基本インタフェース：行動集合・状態集合
    # ------------------------------------------------------------
    def actions(self):
        """
        行動集合 A を返す。
        ここでは [0,1,2,3] の4行動。
        """
        return self.action_space

    def states(self):
        """
        状態集合 S を列挙するジェネレータ。
        盤面上の全セル (y,x) を返す。

        注意：
        - 壁セルも列挙対象に含まれる（(1,1)）点に注意。
          アルゴリズム側で「壁は除外」したい場合はフィルタする設計になる。
        """
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    # ------------------------------------------------------------
    # 遷移関数：s, a -> s'
    # ------------------------------------------------------------
    def next_state(self, state, action):
        """
        遷移関数を決定論的に定義する。
        与えられた state=(y,x) と action から次状態 next_state を返す。

        ここでは移動は 4近傍（上下左右）で、以下の制約がある：
        - 盤面外へ出ようとしたらその場に留まる
        - 壁セルへ入ろうとしたらその場に留まる

        理論的には、この環境は
            P(s'|s,a) が 0/1 しか取らない（確率1で一意に遷移する）
        ので、決定論的MDPの例になる。
        """
        # action -> (dy, dx) の対応表
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # action に対応する移動量を取得
        move = action_move_map[action]

        # 次の座標候補を計算
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state  # y, x を分解

        # 盤面外なら元の state に戻す（移動失敗＝その場に留まる）
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        # 壁なら元の state に戻す（移動失敗＝その場に留まる）
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    # ------------------------------------------------------------
    # 報酬関数：R(s,a,s')
    # ------------------------------------------------------------
    def reward(self, state, action, next_state):
        """
        報酬関数 R(s,a,s') を返す。

        この実装では「次状態 next_state のセルに埋め込まれた値」を報酬とする：
            r = reward_map[next_state]

        つまり、報酬は (s,a) ではなく「到達先の地形」で決まる設計。

        注意：
        - next_state が壁セルになることは next_state() 側で防がれている想定なので、
          reward_map[wall_state] = None が返ることは通常ない。
        """
        return self.reward_map[next_state]

    # ------------------------------------------------------------
    # GymライクなAPI：reset / step
    # ------------------------------------------------------------
    def reset(self):
        """
        エピソード開始：エージェント位置を start_state に戻す。
        返り値は初期状態（観測）としての agent_state。
        """
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        """
        1ステップ進める（環境遷移を実行する）。

        入力：
            action: エージェントが選んだ行動（0..3）

        出力：
            next_state: 遷移後の状態
            reward:     得られた報酬
            done:       終端かどうか（ゴール到達なら True）

        理論的には、ここは MDP の1遷移を生成しており、
        強化学習のデータ (S_t, A_t, R_{t+1}, S_{t+1}) を返している。
        """
        state = self.agent_state

        # 遷移 s,a -> s'
        next_state = self.next_state(state, action)

        # 報酬 r = R(s,a,s')
        reward = self.reward(state, action, next_state)

        # 終端判定：ゴールに到達したら done
        done = next_state == self.goal_state

        # 内部状態を更新
        self.agent_state = next_state

        return next_state, reward, done

    # ------------------------------------------------------------
    # 可視化（レンダリング）
    # ------------------------------------------------------------
    def render_v(self, v=None, policy=None, print_value=True):
        """
        状態価値 V(s) や方策 π(s) を可視化する。

        - v:      dict などで state -> value を渡す想定
        - policy: state -> action（または行動分布）を渡す想定
        - print_value: セルに数値を表示するかどうか

        価値反復・方策評価などで求めた V を人間が確認する目的で使う。
        """
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        """
        行動価値 Q(s,a) を可視化する。

        - q: dict などで (state, action) -> value を渡す想定
        - print_value: 数値を表示するかどうか

        SARSA/Q-learning などで学習した Q を可視化し、
        各状態でどの行動が高いか（greedy action がどれか）を見るのに使う。
        """
        renderer = render_helper.Renderer(
            self.reward_map, self.goal_state, self.wall_state
        )
        renderer.render_q(q, print_value)
