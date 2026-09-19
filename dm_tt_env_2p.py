import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class DeepMindTableTennisEnv2P(gym.Env):
    """嚴格遵循國際桌球 (ITTF) 規則且防作弊的雙人 Self-Play 環境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path("fly_table_2p.xml")
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

        self.ball_body_id = self.model.body("ball").id
        self.ball_geom_id = self.model.geom("ball_geom").id
        self.paddle_1_geom_id = self.model.geom("paddle_1_geom").id
        self.paddle_2_geom_id = self.model.geom("paddle_2_geom").id
        self.table_geom_id = self.model.geom("table_top").id

        # 空氣力學常數
        self.rho, self.r, self.A = 1.225, 0.02, np.pi * (0.02 ** 2)
        self.Cd, self.Cm = 0.47, 0.25

        # ITTF 狀態機變數
        self.last_hitter = None  # 1: P1, 2: P2
        self.bounces_since_hit = 0  # 擊球後的彈跳次數
        self.last_bounce_side = None  # 1: P1檯面 (x<0), 2: P2檯面 (x>0)
        self.is_service_phase = True  # 是否為發球狀態
        self.current_step = 0  # 單回合最大步數保護

    def _apply_aerodynamics(self):
        v = self.data.joint("ball_joint").qvel[0:3]
        w = self.data.joint("ball_joint").qvel[3:6]
        v_mag = np.linalg.norm(v)

        if v_mag > 1e-3:
            f_drag = -0.5 * self.rho * self.Cd * self.A * v_mag * v
            f_magnus = 0.5 * self.rho * self.Cm * self.A * self.r * np.cross(w, v)
            self.data.xfrc_applied[self.ball_body_id, :3] = f_drag + f_magnus
        else:
            self.data.xfrc_applied[self.ball_body_id, :] = 0.0

    def _get_obs(self, agent_id=1):
        ball_pos = self.data.joint("ball_joint").qpos[0:3].copy()
        ball_vel = self.data.joint("ball_joint").qvel[0:3].copy()

        if agent_id == 1:
            fly_pos = [self.data.joint(f"fly_1_{axis}").qpos[0] for axis in ["x", "y", "z"]]
            fly_vel = [self.data.joint(f"fly_1_{axis}").qvel[0] for axis in ["x", "y", "z"]]
            return np.concatenate([fly_pos, fly_vel, ball_pos, ball_vel]).astype(np.float32)
        else:
            fly_pos = [self.data.joint(f"fly_2_{axis}").qpos[0] for axis in ["x", "y", "z"]]
            fly_vel = [self.data.joint(f"fly_2_{axis}").qvel[0] for axis in ["x", "y", "z"]]
            # 對稱鏡像翻轉，讓 P2 視角保持一致
            fly_pos[0], fly_pos[1] = -fly_pos[0], -fly_pos[1]
            fly_vel[0], fly_vel[1] = -fly_vel[0], -fly_vel[1]
            ball_pos[0], ball_pos[1] = -ball_pos[0], -ball_pos[1]
            ball_vel[0], ball_vel[1] = -ball_vel[0], -ball_vel[1]
            return np.concatenate([fly_pos, fly_vel, ball_pos, ball_vel]).astype(np.float32)

    def step_2p(self, action_1, action_2):
        self.current_step += 1
        action_1 = np.asarray(action_1, dtype=np.float64)
        action_2 = np.asarray(action_2, dtype=np.float64)

        # 🛠️ 修復 P2 動作鏡像：P2 模型的向前(+x)/向左(+y)意圖，在世界座標中需反轉方向
        real_action_2 = action_2.copy()
        real_action_2[0] = -action_2[0]  # X 軸：前進/後退反轉
        real_action_2[1] = -action_2[1]  # Y 軸：左右移動反轉
        # 3. 上下 Z 軸 (保持不變)
        real_action_2[2] = action_2[2]

        # 4. 關鍵修復！球拍旋轉角 (必須反轉，否則拍面角度相反！)
        real_action_2[3] = -action_2[3]

        combined_ctrl = np.concatenate([
            np.clip(action_1, -1.0, 1.0),
            np.clip(real_action_2, -1.0, 1.0)  # 修正為 real_action_2
        ])
        self.data.ctrl[:] = combined_ctrl

        hit_p1, hit_p2 = False, False
        paddle1_hit_table, paddle2_hit_table = False, False
        ball_resting_on_table = False
        new_bounce = False
        bounce_side = None

        for _ in range(self.physics_steps_per_action):
            previous_ball_pos = self.data.joint("ball_joint").qpos[0:3].copy()
            self._apply_aerodynamics()
            mujoco.mj_step(self.model, self.data)

            ball_pos_now = self.data.joint("ball_joint").qpos[0:3]
            ball_vel_now = self.data.joint("ball_joint").qvel[0:3]

            on_table = (abs(ball_pos_now[0]) <= self.table_half_x and abs(ball_pos_now[1]) <= self.table_half_y)
            surface = self.table_surface_z + self.ball_radius
            crossed_table = previous_ball_pos[2] >= surface and ball_pos_now[2] < surface
            bounced_this_substep = on_table and crossed_table and ball_vel_now[2] < 0.0

            if bounced_this_substep:
                ball_pos_now[2] = surface
                ball_vel_now[2] = -ball_vel_now[2] * self.table_restitution
                ball_vel_now[0:2] *= 0.96
                new_bounce = True
                bounce_side = 1 if ball_pos_now[0] < 0 else 2

            # 🛠️ 漏洞修復 1: 偵測球是否停在桌上滾動 / 滑行 (鏟球防禦)
            if on_table and ball_pos_now[2] <= (surface + 0.015) and abs(ball_vel_now[2]) < 0.15:
                ball_resting_on_table = True

            # 🛠️ 漏洞修復 2: 偵測球拍與檯面的穿模/砸桌碰撞
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                g1, g2 = con.geom1, con.geom2

                # 球拍觸球檢測
                if self.ball_geom_id in (g1, g2):
                    hit_p1 |= self.paddle_1_geom_id in (g1, g2)
                    hit_p2 |= self.paddle_2_geom_id in (g1, g2)

                # 球拍砸桌/穿模檢測
                if self.table_geom_id in (g1, g2):
                    paddle1_hit_table |= self.paddle_1_geom_id in (g1, g2)
                    paddle2_hit_table |= self.paddle_2_geom_id in (g1, g2)

        reward_1, reward_2 = 0.0, 0.0
        terminated = False

        # 🛑 防作弊處罰 1: 球拍撞桌/穿模懲罰
        if paddle1_hit_table:
            reward_1 -= 15.0
            terminated = True
        if paddle2_hit_table:
            reward_2 -= 15.0
            terminated = True

        # 🛑 防作弊處罰 2: 球停留在桌上滾動 (立刻判定死球)
        if ball_resting_on_table and not terminated:
            terminated = True
            ball_pos_x = self.data.joint("ball_joint").qpos[0]
            if ball_pos_x < 0:
                reward_1 -= 10.0;
                reward_2 += 10.0
            else:
                reward_2 -= 10.0;
                reward_1 += 10.0

        # 🌟 1. ITTF 落檯狀態機
        if new_bounce and not terminated:
            self.bounces_since_hit += 1
            self.last_bounce_side = bounce_side

            if self.is_service_phase:
                if self.last_hitter == 1 and bounce_side == 2 and self.bounces_since_hit == 2:
                    self.is_service_phase = False
                elif self.last_hitter == 2 and bounce_side == 1 and self.bounces_since_hit == 2:
                    self.is_service_phase = False
            else:
                # 正常回合：如果在同一側連續彈跳兩次，判該側輸球
                if self.bounces_since_hit >= 2:
                    terminated = True
                    if bounce_side == 1:
                        reward_1 -= 10.0;
                        reward_2 += 20.0
                    else:
                        reward_2 -= 10.0;
                        reward_1 += 20.0

        # 🌟 2. 稠密獎勵：鼓勵靠近 + 近距離揮拍
        ball_pos = self.data.joint("ball_joint").qpos[0:3]
        paddle1_pos = self.data.geom("paddle_1_geom").xpos
        paddle2_pos = self.data.geom("paddle_2_geom").xpos

        dist_1 = np.linalg.norm(ball_pos - paddle1_pos)
        dist_2 = np.linalg.norm(ball_pos - paddle2_pos)

        reward_1 += (1.0 - np.tanh(dist_1)) * 0.05
        reward_2 += (1.0 - np.tanh(dist_2)) * 0.05

        # 正確計算 3D 全方向揮拍速度
        p1_vel_3d = [self.data.joint(f"fly_1_{a}").qvel[0] for a in ["x", "y", "z"]]
        p2_vel_3d = [self.data.joint(f"fly_2_{a}").qvel[0] for a in ["x", "y", "z"]]
        p1_speed = np.linalg.norm(p1_vel_3d)
        p2_speed = np.linalg.norm(p2_vel_3d)

        if dist_1 < 0.35:
            reward_1 += np.clip(p1_speed, 0.0, 3.0) * 0.1
        if dist_2 < 0.35:
            reward_2 += np.clip(p2_speed, 0.0, 3.0) * 0.1

        # 🌟 3. 處理觸球 (僅限非穿模情況)
        if not terminated:
            if hit_p1:
                self.last_hitter = 1
                self.bounces_since_hit = 0
                self.last_bounce_side = None
                reward_1 += 10.0

            if hit_p2:
                self.last_hitter = 2
                self.bounces_since_hit = 0
                self.last_bounce_side = None
                reward_2 += 10.0

        # 🌟 4. 出界 / 沒接到球 (落地死球判定)
        if not terminated and (ball_pos[2] < 0.1 or abs(ball_pos[1]) > 1.1 or abs(ball_pos[0]) > 1.6):
            terminated = True
            if self.last_hitter == 1:
                if self.bounces_since_hit == 1 and self.last_bounce_side == 2:
                    reward_1 += 20.0;
                    reward_2 -= 10.0
                else:
                    reward_1 -= 10.0;
                    reward_2 += 20.0
            elif self.last_hitter == 2:
                if self.bounces_since_hit == 1 and self.last_bounce_side == 1:
                    reward_2 += 20.0;
                    reward_1 -= 10.0
                else:
                    reward_2 -= 10.0;
                    reward_1 += 20.0
            else:
                if ball_pos[0] < 0:
                    reward_1 -= 10.0
                else:
                    reward_2 -= 10.0

        # 🌟 5. 強制最大步數限制 (防止卡死)
        truncated = self.current_step >= 500

        if self.render_mode == "human":
            self.render()

        return self._get_obs(agent_id=1), self._get_obs(agent_id=2), reward_1, reward_2, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 1. 狀態重置：開局尚未有人擊球
        self.last_hitter = None
        self.bounces_since_hit = 0
        self.last_bounce_side = None
        self.is_service_phase = True
        self.current_step = 0

        ball_qpos = self.data.joint("ball_joint").qpos
        ball_qvel = self.data.joint("ball_joint").qvel

        # 2. 隨機決定由哪一側開球 (僅改變球的發射位置與速度方向)
        if self.np_random.random() > 0.5:
            ball_qpos[0], ball_qvel[0] = 0.70, -2.0  # 從 P2 側發出
        else:
            ball_qpos[0], ball_qvel[0] = -0.70, 2.0  # 從 P1 側發出

        ball_qpos[1] = self.np_random.uniform(-0.02, 0.02)
        ball_qpos[2] = 0.75
        ball_qvel[1] = self.np_random.uniform(-0.02, 0.02)
        ball_qvel[2] = -2.0

        # 3. 球拍初始位置
        self.data.joint("fly_1_x").qpos[0] = -0.8
        self.data.joint("fly_1_z").qpos[0] = 0.5
        self.data.joint("fly_2_x").qpos[0] = 0.8
        self.data.joint("fly_2_z").qpos[0] = 0.5

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(agent_id=1), self._get_obs(agent_id=2), {}

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