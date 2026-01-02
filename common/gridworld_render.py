import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Renderer:
    """
    GridWorld の状態価値 V(s) や行動価値 Q(s,a) を可視化するためのレンダラ。

    強化学習では、学習した結果が「直感的に正しそうか」を確認するのがとても重要であり、
    グリッドワールドのような小規模環境では可視化が強力なデバッグ手段になる。

    このレンダラは大きく2種類の表示を提供する：
    - render_v:  状態価値 V(s) をセルごとにヒートマップ表示し、必要なら方策（矢印）も描く
    - render_q:  各セルを4つの三角形に分割して、行動価値 Q(s,a) を方向別に色付けして表示する

    なお、Matplotlib の座標系と「グリッド上の (y,x)」の上下方向が逆になりがちなので、
    render_v では np.flipud を使い、見た目が自然（上が y=0）になるよう調整している。
    """

    def __init__(self, reward_map, goal_state, wall_state):
        # reward_map: 盤面の地形。各セルの報酬（0, +1, -1, None=壁など）を持つ
        self.reward_map = reward_map

        # goal_state: ゴールの座標 (y,x)
        self.goal_state = goal_state

        # wall_state: 壁（侵入不可）の座標 (y,x)
        self.wall_state = wall_state

        # 盤面サイズ（行数=ys, 列数=xs）
        self.ys = len(self.reward_map)
        self.xs = len(self.reward_map[0])

        # Matplotlib の Figure / Axes を保持する（必要なら再利用できる）
        self.ax = None
        self.fig = None

        # first_flg はアニメーション用途などで「初回だけ初期化する」ために置かれがち。
        # 現在のコードでは実質使われていないが、拡張の余地として残っている。
        self.first_flg = True

    def set_figure(self, figsize=None):
        """
        描画用の Figure と Axes を作り、グリッドの枠線・目盛りを設定する。

        Matplotlib は標準で「左下が原点」で y が上向きに増えるが、
        グリッドワールドでは (y,x) を行列インデックスとして扱い、
        y=0 が上端であることが多い。
        そのため render_v 側で np.flipud を使って上下反転して表示している。

        ここでは座標軸ラベルを消し、セル境界が分かるように grid を表示している。
        """
        fig = plt.figure(figsize=figsize)
        self.ax = fig.add_subplot(111)
        ax = self.ax

        # 既存描画があればクリア（同じ Axes を使い回す設計）
        ax.clear()

        # 目盛りラベルは不要なので非表示
        ax.tick_params(
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )

        # グリッド線用に ticks を設定（セル境界に線を引く）
        ax.set_xticks(range(self.xs))
        ax.set_yticks(range(self.ys))

        # 表示範囲（グリッドの外枠）
        ax.set_xlim(0, self.xs)
        ax.set_ylim(0, self.ys)

        # グリッド線表示
        ax.grid(True)

    def render_v(self, v=None, policy=None, print_value=True):
        """
        状態価値 V(s)（任意）と方策 π(s)（任意）を描画する。

        引数：
        - v: dict or ndarray
            - dict の場合：キーが state=(y,x)、値が V(s)
            - ndarray の場合：reward_map と同形状の 2次元配列
        - policy: dict
            - policy[state] が {action: prob} の辞書（行動分布）
            - ここでは確率最大の行動（greedy）に矢印を描く
        - print_value: セル上に数値として V(s) を表示するか

        表示内容：
        - v があれば、ヒートマップ（赤=低い, 白=中間, 緑=高い）で色付け
        - 報酬セル（+1,-1など）には "R ..." の注記
        - policy があれば、最確率行動に対応する矢印を表示（複数同率なら複数矢印）
        - wall_state は灰色のブロックで塗りつぶし
        """
        self.set_figure()

        ys, xs = self.ys, self.xs
        ax = self.ax

        # -----------------------------
        # V(s) のヒートマップ表示
        # -----------------------------
        if v is not None:
            # カラーマップを自作：低い=赤、中間=白、高い=緑
            # ※ v の符号・スケールで色の出方が変わるため、後で vmin/vmax を調整する。
            color_list = ["red", "white", "green"]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "colormap_name", color_list
            )

            # v が dict の場合は ndarray に変換して扱いやすくする
            # （可視化の都合上、盤面上の (y,x) を配列インデックスとして配置）
            v_dict = v
            v = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v[state] = value

            # 色のスケールを決める。
            # vmax/vmin を対称にしておくと「0を中心に正負が見やすい」ヒートマップになる。
            vmax, vmin = v.max(), v.min()
            vmax = max(vmax, abs(vmin))  # |min| と max の大きい方を採用
            vmin = -1 * vmax

            # 極端に小さい値しかないと色がほぼ白になって見えないので下限を設ける
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin

            # np.flipud(v) により上下反転して描画。
            # これにより (0,0) が左上として見えるようになり、グリッドワールドの直感に合う。
            ax.pcolormesh(np.flipud(v), cmap=cmap, vmin=vmin, vmax=vmax)

        # -----------------------------
        # セルごとの注記（報酬、V値、方策矢印、壁）
        # -----------------------------
        for y in range(ys):
            for x in range(xs):
                state = (y, x)

                # reward_map に埋め込まれた報酬（ゴール/罠など）を表示
                r = self.reward_map[y, x]
                if r != 0 and r is not None:
                    txt = "R " + str(r)
                    if state == self.goal_state:
                        txt = txt + " (GOAL)"
                    # テキスト描画座標は matplotlib の座標系に合わせるため y を反転している
                    ax.text(x + 0.1, ys - y - 0.9, txt)

                # V(s) の数値表示（壁セルは除外）
                if (v is not None) and state != self.wall_state:
                    if print_value:
                        # 盤面が大きいと数字が重なりやすいのでオフセットを切替
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v.shape[0] > 7:
                            key = 1
                        offset = offsets[key]
                        ax.text(
                            x + offset[0],
                            ys - y + offset[1],
                            "{:12.2f}".format(v[y, x]),
                        )

                # 方策（policy）が与えられた場合、確率最大の行動に矢印を描く
                if policy is not None and state != self.wall_state:
                    actions = policy[state]

                    # 最大確率を持つ行動を抽出（同率最大が複数ある場合は複数矢印を描く）
                    max_actions = [
                        kv[0]
                        for kv in actions.items()
                        if kv[1] == max(actions.values())
                    ]

                    # 行動ID -> 矢印文字
                    arrows = ["↑", "↓", "←", "→"]
                    # 矢印の位置を少しズラす（複数矢印のときに見やすい）
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]

                    for action in max_actions:
                        # ゴールには矢印を出しても意味が薄い（終端）のでスキップ
                        if state == self.goal_state:
                            continue
                        arrow = arrows[action]
                        offset = offsets[action]
                        ax.text(x + 0.45 + offset[0], ys - y - 0.5 + offset[1], arrow)

                # 壁セルは灰色の四角で塗りつぶす
                if state == self.wall_state:
                    ax.add_patch(
                        plt.Rectangle((x, ys - y - 1), 1, 1, fc=(0.4, 0.4, 0.4, 1.0))
                    )

        plt.show()

    def render_q(self, q, show_greedy_policy=True):
        """
        行動価値 Q(s,a) を可視化する。

        基本アイデア：
        - 各セルを4つの三角形に分割し、それぞれが action (UP/DOWN/LEFT/RIGHT) を表す。
        - Q(s,a) の値に応じて色を付け、どの行動が良いかを視覚化する。
        - さらに show_greedy_policy=True なら、各状態で argmax_a Q(s,a) を取った決定的方策を作り、
          render_v を使って矢印として表示する。

        入力 q の想定形式：
        - q[(state, action)] = value の辞書（state=(y,x), action=0..3）

        注意：
        - 壁セルは灰色、ゴールセルは（意図として）緑に塗りたいが、
          現状 `elif state in self.goal_state:` はバグの可能性が高い。
          goal_state は (y,x) のタプルなので `state == self.goal_state` が正しい条件になりやすい。
        """
        self.set_figure()

        ys, xs = self.ys, self.xs
        ax = self.ax
        action_space = [0, 1, 2, 3]

        # -----------------------------
        # 色スケール（vmin/vmax）の決定
        # -----------------------------
        # Q の最大値と最小値を取り、対称レンジにすることで正負が見やすい。
        qmax, qmin = max(q.values()), min(q.values())
        qmax = max(qmax, abs(qmin))
        qmin = -1 * qmax
        qmax = 1 if qmax < 1 else qmax
        qmin = -1 if qmin > -1 else qmin

        # カラーマップ（低い=赤、中間=白、高い=緑）
        color_list = ["red", "white", "green"]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "colormap_name", color_list
        )

        # -----------------------------
        # 各セル・各行動の三角形を描画
        # -----------------------------
        for y in range(ys):
            for x in range(xs):
                for action in action_space:
                    state = (y, x)

                    # 報酬セルの注記（+1/-1など）
                    r = self.reward_map[y, x]
                    if r != 0 and r is not None:
                        txt = "R " + str(r)
                        if state == self.goal_state:
                            txt = txt + " (GOAL)"
                        ax.text(x + 0.05, ys - y - 0.95, txt)

                    # ゴールセルは通常「終端」で、そこでの Q(s,a) を描く意味が薄いのでスキップ
                    if state == self.goal_state:
                        continue

                    # Matplotlib 座標系での左下座標（tx,ty）を計算
                    tx, ty = x, ys - y - 1

                    # セルを4三角形に分割して action ごとの領域を定義
                    # 各 action の三角形は「頂点3つ（x,y）」で指定する。
                    action_map = {
                        # UP：セル上側の三角形
                        0: ((0.5 + tx, 0.5 + ty), (tx + 1, ty + 1), (tx, ty + 1)),
                        # DOWN：セル下側の三角形
                        1: ((tx, ty), (tx + 1, ty), (tx + 0.5, ty + 0.5)),
                        # LEFT：セル左側の三角形
                        2: ((tx, ty), (tx + 0.5, ty + 0.5), (tx, ty + 1)),
                        # RIGHT：セル右側の三角形
                        3: ((0.5 + tx, 0.5 + ty), (tx + 1, ty), (tx + 1, ty + 1)),
                    }

                    # Q値の数値表示位置（セル内でのテキスト配置）
                    offset_map = {
                        0: (0.1, 0.8),
                        1: (0.1, 0.1),
                        2: (-0.2, 0.4),
                        3: (0.4, 0.4),
                    }

                    # 壁セルは灰色塗りつぶし
                    if state == self.wall_state:
                        ax.add_patch(
                            plt.Rectangle((tx, ty), 1, 1, fc=(0.4, 0.4, 0.4, 1.0))
                        )
                    # ゴールセルの塗りつぶし（意図としてはここで緑にしたい）
                    # 注意：`state in self.goal_state` はタプル membership 判定になりやすく誤りの可能性が高い。
                    #       正しくは `state == self.goal_state` が自然。
                    elif state in self.goal_state:
                        ax.add_patch(
                            plt.Rectangle((tx, ty), 1, 1, fc=(0.0, 1.0, 0.0, 1.0))
                        )
                    else:
                        # q は (state, action) をキーにした辞書を想定
                        tq = q[(state, action)]

                        # 値を 0.0-1.0 に正規化して色を決める
                        # ここでは qmax によるスケーリングで対称レンジを想定している。
                        color_scale = 0.5 + (tq / qmax) / 2  # normalize: 0.0-1.0

                        # 三角形パッチを追加して色付け
                        poly = plt.Polygon(action_map[action], fc=cmap(color_scale))
                        ax.add_patch(poly)

                        # Q値を数値で表示
                        offset = offset_map[action]
                        ax.text(tx + offset[0], ty + offset[1], "{:12.2f}".format(tq))

        plt.show()

        # -----------------------------
        # greedy 方策の可視化（Qから導出）
        # -----------------------------
        if show_greedy_policy:
            # policy[state] = {action: prob} の形式を作る（ここでは決定的：argmax に確率1）
            policy = {}
            for y in range(self.ys):
                for x in range(self.xs):
                    state = (y, x)

                    # 4行動の Q を取り出し、最大の行動を選ぶ
                    qs = [q[state, action] for action in range(4)]  # action_size
                    max_action = np.argmax(qs)

                    # 決定的方策：最大行動だけ確率1
                    probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
                    probs[max_action] = 1
                    policy[state] = probs

            # V は描かず（None）、方策矢印だけ描画する
            self.render_v(None, policy)
