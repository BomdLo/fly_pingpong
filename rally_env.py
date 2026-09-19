"""Stable two-player environment for learning actual ping-pong rallies.

MuJoCo simulates the two flies and paddles.  The 2.7 g ball uses deterministic
event collisions, avoiding the energy spikes that compliant rigid contacts can
produce at a 2 ms timestep.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class RallyTableTennisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("fly_table_2p.xml")
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 30
        self.data = mujoco.MjData(self.model)
        self.render_mode, self.viewer = render_mode, None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        self.physics_steps_per_action = 10
        self.dt = self.model.opt.timestep
        self.ball_radius = 0.02
        self.table_surface_z = 0.40
        self.table_half_x, self.table_half_y = 1.37, 0.7625
        self.net_top_z = 0.552
        self.table_restitution = 0.84
        self.paddle_radius, self.paddle_half_thickness = 0.10, 0.008
        self.max_steps = 800

        self.ball_body_id = self.model.body("ball").id
        self.paddle_geom_ids = {1: self.model.geom("paddle_1_geom").id,
                                2: self.model.geom("paddle_2_geom").id}
        self._reset_rally_state()

    @staticmethod
    def opponent(player):
        return 2 if player == 1 else 1

    def _reset_rally_state(self):
        self.server = 2
        self.phase = "serve_first_bounce"
        self.expected_hitter = None
        self.last_hitter = None
        self.rally_hits = 0
        self.current_step = 0
        self.hit_cooldown = {1: 0.0, 2: 0.0}

    def _apply_aerodynamics(self):
        velocity = self.data.joint("ball_joint").qvel[0:3]
        speed = np.linalg.norm(velocity)
        if speed <= 1e-4:
            self.data.xfrc_applied[self.ball_body_id, :] = 0.0
            return
        area = np.pi * self.ball_radius ** 2
        drag = -0.5 * 1.225 * 0.47 * area * speed * velocity
        self.data.xfrc_applied[self.ball_body_id, :] = 0.0
        self.data.xfrc_applied[self.ball_body_id, :3] = drag

    def _get_obs(self, player):
        ball_pos = self.data.joint("ball_joint").qpos[0:3].copy()
        ball_vel = self.data.joint("ball_joint").qvel[0:3].copy()
        fly_pos = np.array([self.data.joint(f"fly_{player}_{axis}").qpos[0]
                            for axis in ("x", "y", "z")])
        fly_vel = np.array([self.data.joint(f"fly_{player}_{axis}").qvel[0]
                            for axis in ("x", "y", "z")])
        if player == 2:
            # Both policies see themselves on the left, returning toward +X.
            fly_pos[0:2] *= -1.0
            fly_vel[0:2] *= -1.0
            ball_pos[0:2] *= -1.0
            ball_vel[0:2] *= -1.0
        return np.concatenate((fly_pos, fly_vel, ball_pos, ball_vel)).astype(np.float32)

    def _paddle_hit(self, player, ball_pos, ball_vel):
        if self.hit_cooldown[player] > 0.0:
            return False
        geom = self.data.geom(self.paddle_geom_ids[player])
        paddle_pos = geom.xpos
        rotation = geom.xmat.reshape(3, 3)
        normal = rotation[:, 2]  # cylinder axis after euler="0 90 0"
        offset = ball_pos - paddle_pos
        normal_distance = float(np.dot(offset, normal))
        tangential_distance = np.linalg.norm(offset - normal_distance * normal)
        approaching = ball_vel[0] < -0.1 if player == 1 else ball_vel[0] > 0.1
        return (approaching
                and abs(normal_distance) <= self.ball_radius + self.paddle_half_thickness
                and tangential_distance <= self.paddle_radius + self.ball_radius)

    def _launch_from_paddle(self, player):
        ball_pos = self.data.joint("ball_joint").qpos[0:3]
        ball_vel = self.data.joint("ball_joint").qvel[0:3]
        paddle_pos = self.data.geom(self.paddle_geom_ids[player]).xpos
        direction = 1.0 if player == 1 else -1.0
        fly_y_velocity = self.data.joint(f"fly_{player}_y").qvel[0]
        fly_z_velocity = self.data.joint(f"fly_{player}_z").qvel[0]
        ball_pos[:] = paddle_pos
        ball_pos[0] += direction * (self.ball_radius + self.paddle_half_thickness + 0.004)
        ball_vel[0] = direction * 2.25
        ball_vel[1] = np.clip(fly_y_velocity * 0.20, -0.45, 0.45)
        ball_vel[2] = 2.05 + np.clip(fly_z_velocity * 0.15, -0.25, 0.25)

    def _finish(self, loser, reason, rewards):
        winner = self.opponent(loser)
        rewards[loser] -= 20.0
        rewards[winner] += 20.0
        return True, {"winner": winner, "loser": loser, "reason": reason,
                      "rally_hits": self.rally_hits}

    def step_2p(self, action_1, action_2):
        self.current_step += 1
        action_1 = np.clip(np.asarray(action_1, dtype=np.float64), -1.0, 1.0)
        action_2 = np.clip(np.asarray(action_2, dtype=np.float64), -1.0, 1.0)
        # Agent 2 acts in mirrored coordinates.
        world_action_2 = action_2.copy()
        world_action_2[[0, 1, 3]] *= -1.0
        self.data.ctrl[:] = np.concatenate((action_1, world_action_2))

        bounce_sides, hit_players, net_fault = [], [], False
        for _ in range(self.physics_steps_per_action):
            previous_pos = self.data.joint("ball_joint").qpos[0:3].copy()
            self._apply_aerodynamics()
            mujoco.mj_step(self.model, self.data)
            ball_pos = self.data.joint("ball_joint").qpos[0:3]
            ball_vel = self.data.joint("ball_joint").qvel[0:3]
            for player in (1, 2):
                self.hit_cooldown[player] = max(0.0, self.hit_cooldown[player] - self.dt)

            crossed_net = previous_pos[0] * ball_pos[0] <= 0.0
            if crossed_net and ball_pos[2] < self.net_top_z + self.ball_radius:
                net_fault = True

            on_table = abs(ball_pos[0]) <= self.table_half_x and abs(ball_pos[1]) <= self.table_half_y
            surface = self.table_surface_z + self.ball_radius
            if on_table and previous_pos[2] >= surface and ball_pos[2] < surface and ball_vel[2] < 0.0:
                ball_pos[2] = surface
                ball_vel[2] = -ball_vel[2] * self.table_restitution
                ball_vel[0:2] *= 0.96
                bounce_sides.append(1 if ball_pos[0] < 0.0 else 2)

            for player in (1, 2):
                if self._paddle_hit(player, ball_pos, ball_vel):
                    self._launch_from_paddle(player)
                    self.hit_cooldown[player] = 0.10
                    hit_players.append(player)
                    break

        rewards = {1: 0.0, 2: 0.0}
        terminated, info = False, {"rally_hits": self.rally_hits}
        if net_fault:
            loser = self.last_hitter if self.last_hitter is not None else self.server
            terminated, info = self._finish(loser, "net", rewards)

        if not terminated:
            for side in bounce_sides:
                if self.phase == "serve_first_bounce":
                    if side != self.server:
                        terminated, info = self._finish(self.server, "bad_serve", rewards)
                        break
                    self.phase = "serve_second_bounce"
                elif self.phase == "serve_second_bounce":
                    receiver = self.opponent(self.server)
                    if side != receiver:
                        terminated, info = self._finish(self.server, "bad_serve", rewards)
                        break
                    self.phase = "rally"
                    self.expected_hitter = receiver
                elif self.expected_hitter is None:
                    target = self.opponent(self.last_hitter)
                    if side != target:
                        terminated, info = self._finish(self.last_hitter, "wrong_side", rewards)
                        break
                    self.expected_hitter = target
                    rewards[self.last_hitter] += 3.0
                else:
                    terminated, info = self._finish(self.expected_hitter, "second_bounce", rewards)
                    break

        if not terminated:
            for player in hit_players:
                if self.phase != "rally" or player != self.expected_hitter:
                    terminated, info = self._finish(player, "early_or_wrong_hit", rewards)
                    break
                self.last_hitter = player
                self.expected_hitter = None
                self.rally_hits += 1
                rewards[player] += 8.0 + min(self.rally_hits, 20) * 0.25

        ball_pos = self.data.joint("ball_joint").qpos[0:3]
        if not terminated and (ball_pos[2] < 0.08 or abs(ball_pos[1]) > 1.05 or abs(ball_pos[0]) > 1.60):
            loser = self.expected_hitter or self.last_hitter or self.server
            terminated, info = self._finish(loser, "out", rewards)

        # Dense guidance is only given to the player whose turn it is.
        if not terminated and self.expected_hitter is not None:
            paddle_pos = self.data.geom(self.paddle_geom_ids[self.expected_hitter]).xpos
            rewards[self.expected_hitter] += 0.10 * (1.0 - np.tanh(np.linalg.norm(ball_pos - paddle_pos)))

        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            info = {"winner": None, "reason": "time_limit", "rally_hits": self.rally_hits}
        if self.render_mode == "human":
            self.render()
        return (self._get_obs(1), self._get_obs(2), rewards[1], rewards[2],
                terminated, truncated, info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_rally_state()
        self.server = 2 if self.np_random.random() > 0.5 else 1
        ball_pos = self.data.joint("ball_joint").qpos
        ball_vel = self.data.joint("ball_joint").qvel
        server_side = 1.0 if self.server == 2 else -1.0
        ball_pos[0], ball_pos[1], ball_pos[2] = server_side * 0.70, self.np_random.uniform(-0.02, 0.02), 0.75
        ball_vel[0], ball_vel[1], ball_vel[2] = -server_side * 2.0, self.np_random.uniform(-0.02, 0.02), -2.0
        ball_vel[3:6] = 0.0
        for player, x in ((1, -0.80), (2, 0.80)):
            self.data.joint(f"fly_{player}_x").qpos[0] = x
            self.data.joint(f"fly_{player}_y").qpos[0] = 0.0
            self.data.joint(f"fly_{player}_z").qpos[0] = 0.50
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(1), self._get_obs(2), {}

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
