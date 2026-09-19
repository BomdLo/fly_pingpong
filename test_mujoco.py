import time
import torch
from stable_baselines3 import PPO
from dm_tt_env import DeepMindTableTennisEnv


def test_mujoco_vis():
    # 開啟 human 模式啟動 3D 視窗
    env = DeepMindTableTennisEnv(render_mode="human")

    device = "cpu"
    model = PPO.load("fly_pingpong_pro_model", device=device)

    print("🏓 啟動 MuJoCo 3D 視覺化測試視窗...")

    for episode in range(1, 20):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            env.render()
            time.sleep(0.01)  # 控制播放速率

        print(f"🏁 第 {episode} 局結束 | 總生存步數: {step}")

    env.close()


if __name__ == "__main__":
    test_mujoco_vis()