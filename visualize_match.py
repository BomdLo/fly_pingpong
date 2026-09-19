import time
import os
import numpy as np
from stable_baselines3 import PPO
from rally_env import RallyTableTennisEnv


def run_visualization(model_p1_path, model_p2_path=None, num_episodes=5, fps=50):
    """
    3D 視覺化驗收腳本
    :param model_p1_path: P1 載入的模型路徑
    :param model_p2_path: P2 載入的模型路徑 (若為 None 則與 P1 同款對打)
    :param num_episodes: 播放總回合數
    :param fps: 渲染目標幀率
    """
    if not os.path.exists(model_p1_path + ".zip") and not os.path.exists(model_p1_path):
        print(f"❌ 找不到模型檔案: {model_p1_path}")
        return

    print("🎮 正在初始化 MuJoCo 3D 渲染視窗...")
    env = RallyTableTennisEnv(render_mode="human")

    # 載入模型
    print(f"📦 載入 P1 模型: {model_p1_path}")
    model_p1 = PPO.load(model_p1_path)

    if model_p2_path and model_p2_path != model_p1_path:
        print(f"📦 載入 P2 模型 (跨世代對決): {model_p2_path}")
        model_p2 = PPO.load(model_p2_path)
    else:
        print("⚔️ P2 使用與 P1 相同的最新模型 (镜像對攻)")
        model_p2 = model_p1

    frame_delay = 1.0 / fps

    for episode in range(1, num_episodes + 1):
        obs_1, obs_2, _ = env.reset()
        done = False
        step_count = 0
        p1_total_reward = 0.0
        p2_total_reward = 0.0

        print(f"\n==================== 🎾 [第 {episode}/{num_episodes} 回合開始] ====================")

        while not done:
            # 使用 deterministic=True 檢視模型的最佳策略發揮（非隨機採樣）
            action_1, _ = model_p1.predict(obs_1, deterministic=True)
            action_2, _ = model_p2.predict(obs_2, deterministic=True)

            # 執行一步環境模擬
            obs_1, obs_2, r1, r2, terminated, truncated, info = env.step_2p(action_1, action_2)

            p1_total_reward += r1
            p2_total_reward += r2
            step_count += 1
            done = terminated or truncated

            # 控制渲染速度，避免 GPU 跑太快看不清
            time.sleep(frame_delay)

        # 判斷死球原因與回合數據
        duration_sec = step_count * 0.02  # 每步 0.02 秒物理時間
        print(f"🏁 [回合結束]")
        print(f"⏱️ 總堅持步數 : {step_count} 步 ({duration_sec:.2f} 秒)")
        print(f"📊 P1 累積得分 : {p1_total_reward:.2f} | P2 累積得分 : {p2_total_reward:.2f}")

    print("\n✅ 驗收展示完成，關閉視窗。")



if __name__ == "__main__":
    # 預設進行 v5 最新模型的镜像對打
    LATEST_MODEL = "self_play_model"

    # 若想測試跨世代對決，可填入舊模型檔名，例如 "self_play_model_v1"
    OLD_MODEL = None

    run_visualization(
        model_p1_path=LATEST_MODEL,
        model_p2_path=OLD_MODEL,
        num_episodes=5,
        fps=50  # 可調慢至 20~30 FPS 觀察慢動作細節
    )
