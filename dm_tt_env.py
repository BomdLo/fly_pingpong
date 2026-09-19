import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class DeepMindTableTennisEnv(gym.Env):
    """結合 DeepMind 物理求解器與正則乒乓球規則的 Gymnasium 環境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        # 1. 載入物理模型
        self.model = mujoco.MjModel.from_xml_path("fly_table.xml")

        # 覆寫為 DeepMind 黃金碰撞與 Newton 求解器 (徹底消除數值暴衝)
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 30
        self.model.opt.tolerance = 1e-10
        self.data = mujoco.MjData(self.model)

        self.physics_steps_per_action = 10
        self.ball_radius = 0.02
        self.table_surface_z = 0.40
        self.table_half_x = 1.37
        self.table_half_y = 0.7625
        self.table_restitution = 0.84

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # 效能優化：預先快取 Geom 與 Body 整數 ID
        self.ball_body_id = self.model.body("ball").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        self.paddle_geom_id = self.model.geom("paddle_geom").id
        self.table_geom_id = self.model.geom("table_top").id

        # 空氣力學常數 (標準 2.7g 桌球)
        self.rho = 1.225
        self.r = 0.02
        self.A = np.pi * (self.r ** 2)
        self.Cd = 0.47
        self.Cm = 0.25

        # 狀態機追蹤變數
        self.has_been_hit = False
        self.server_bounce = False
        self.fly_bounces = 0
        self.was_touching_table = False
        self.has_crossed_net = False  # 記錄擊球後是否成功過網

    def _apply_aerodynamics(self):
        """計算空氣阻力與馬格努斯揚力"""
        v = self.data.joint("ball_joint").qvel[0:3]
        w = self.data.joint("ball_joint").qvel[3:6]
        v_mag = np.linalg.norm(v)

        if v_mag > 1e-3:
            f_drag = -0.5 * self.rho * self.Cd * self.A * v_mag * v
            f_magnus = 0.5 * self.rho * self.Cm * self.A * self.r * np.cross(w, v)
            self.data.xfrc_applied[self.ball_body_id, :3] = f_drag + f_magnus
        else:
            self.data.xfrc_applied[self.ball_body_id, :] = 0.0

    def _get_obs(self):
        fly_pos = [self.data.joint(f"fly_{axis}").qpos[0] for axis in ["x", "y", "z"]]
        fly_vel = [self.data.joint(f"fly_{axis}").qvel[0] for axis in ["x", "y", "z"]]
        ball_pos = self.data.joint("ball_joint").qpos[0:3]
        ball_vel = self.data.joint("ball_joint").qvel[0:3]
        return np.concatenate([fly_pos, fly_vel, ball_pos, ball_vel]).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)

        hit_paddle = False
        new_bounce = False
        for _ in range(self.physics_steps_per_action):
            previous_ball_pos = self.data.joint("ball_joint").qpos[0:3].copy()
            self._apply_aerodynamics()
            mujoco.mj_step(self.model, self.data)

            ball_pos_now = self.data.joint("ball_joint").qpos[0:3]
            ball_vel_now = self.data.joint("ball_joint").qvel[0:3]
            on_table = (
                    abs(ball_pos_now[0]) <= self.table_half_x
                    and abs(ball_pos_now[1]) <= self.table_half_y
            )
            surface = self.table_surface_z + self.ball_radius
            crossed_table = previous_ball_pos[2] >= surface and ball_pos_now[2] < surface
            bounced_this_substep = on_table and crossed_table and ball_vel_now[2] < 0.0
            if bounced_this_substep:
                ball_pos_now[2] = surface
                ball_vel_now[2] = -ball_vel_now[2] * self.table_restitution
                ball_vel_now[0:2] *= 0.96

            touching_table = False
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                g1, g2 = con.geom1, con.geom2
                if self.ball_geom_id in (g1, g2):
                    hit_paddle |= self.paddle_geom_id in (g1, g2)
                    touching_table |= self.table_geom_id in (g1, g2)

            new_bounce |= bounced_this_substep
            self.was_touching_table = touching_table

        obs = self._get_obs()
        fly_pos, ball_pos = obs[0:3], obs[6:9]
        ball_vel = obs[9:12]
        dist = np.linalg.norm(fly_pos - ball_pos)

        # 距離引導獎勵
        reward = (1.0 - np.tanh(dist)) * 0.2
        terminated = False

        # --- 過網中繼獎勵 ---
        # 如果已經擊球，且球的 X 座標大於 0 (跨過球網)，且這回合還沒領過過網獎勵
        if getattr(self, 'has_been_hit', False) and ball_pos[0] > 0.0 and not getattr(self, 'has_crossed_net', False):
            self.has_crossed_net = True
            reward +=300.0  # 🌟 鼓勵把球打高、打過網的中繼大獎！

        # 1. 落檯與規則判定
        if new_bounce:
            if getattr(self, 'has_been_hit', False):
                if ball_pos[0] > 0.0:
                    reward += 1000.0  # 🎯 完美目標：過網並落檯成功！
                    terminated = True
                else:
                    reward -= 100.0  # ❌ 懲罰：打在自家半場 (掛網或往下砸)
                    terminated = True
            else:
                # 發球階段
                if ball_pos[0] > 0.0:
                    if not getattr(self, 'server_bounce', False):
                        self.server_bounce = True
                    else:
                        terminated = True
                elif ball_pos[0] < 0.0:
                    if not getattr(self, 'server_bounce', False):
                        terminated = True
                    else:
                        self.fly_bounces += 1
                        if getattr(self, 'fly_bounces', 0) > 1:
                            terminated = True

        # 2. 擊球邏輯：嚴格要求先落檯才能揮擊
        if hit_paddle and not getattr(self, 'has_been_hit', False):
            if getattr(self, 'fly_bounces', 0) >= 1:
                # 合法擊球：已經在己方檯面彈跳過
                self.has_been_hit = True
                reward += 100.0
                # 🌟 新增：安全版「麵包屑」獎勵（引導方向，但不讓它洗分）
                vx = ball_vel[0]  # 往前速度
                vz = ball_vel[2]  # 往上速度

                if vx > 0:
                    reward += min(vx * 10.0, 30.0)  # 鼓勵往前，但最多只給 30 分
                else:
                    reward -= 100.0
                if vz > 0:
                    reward += min(vz * 10.0, 30.0)  # 鼓勵往上，但最多只給 30 分


            else:
                # 攔截犯規：球根本還沒落檯就偷跑去碰球
                reward -= 100.0
                terminated = True

        # 3. 邊界與落地判定 (出界懲罰)
        if ball_pos[2] < 0.1 or abs(ball_pos[1]) > 1.1 or abs(ball_pos[0]) > 1.6:
            if getattr(self, 'has_been_hit', False):
                if ball_pos[0] > 0.0:
                    # ❌ 打過網了，但直接飛出界外 (沒有碰到對方桌子) -> 全壘打重罰！
                    reward -= 100.0
                elif not getattr(self, 'has_crossed_net', False):
                    # ❌ 沒過網就出界
                    reward -= 100.0
            terminated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.has_been_hit = False
        self.server_bounce = False
        self.fly_bounces = 0
        self.was_touching_table = False
        self.has_crossed_net = False  # 記錄擊球後是否成功過網

        ball_qpos = self.data.joint("ball_joint").qpos
        ball_qvel = self.data.joint("ball_joint").qvel
        ball_qpos[0] = 0.70
        ball_qpos[1] = self.np_random.uniform(-0.02, 0.02)
        ball_qpos[2] = 0.75

        ball_qvel[0] = -2.0
        ball_qvel[1] = self.np_random.uniform(-0.02, 0.02)
        ball_qvel[2] = -2.0
        ball_qvel[3:6] = 0.0

        self.data.joint("fly_x").qpos[0] = -0.8
        self.data.joint("fly_y").qpos[0] = 0.0
        self.data.joint("fly_z").qpos[0] = 0.5

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def render(self):
        if self.viewer is None and self.render_mode == "human":
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None