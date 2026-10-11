🏓 3D Fly PingPong RL (飛行器桌球強化學習)

本專案是一個基於 MuJoCo 物理引擎與 PPO (Proximal Policy Optimization) 算法的 3D 雙人飛行器桌球對打環境。透過 Self-Play (自我對弈) 策略，訓練具有 3D 空間感知、精準控球與高速強攻能力的 AI 選手。

🌟 專案亮點 (Key Features)

3D 物理動態模擬：基於 MuJoCo 精確計算球體反彈、旋轉摩擦與飛行器關節阻尼（$F = ma$ 精準轉化）。

完全對稱對弈環境：包含 P1（左半場）與 P2（右半場）雙飛行器，具備 3D 滑軌移動（X, Y, Z）與拍面轉向（Hinge）共 4 自由度。

Self-Play 自我對弈：支援模型自動快照儲存與歷史版本抽樣對戰，促使策略演化出進攻與防守打法。

真實物理調教：經過精密驗證的馬達推力與滑軌阻尼設定（gear="7.0", damping="0.08"），消除動作暴衝抖動與瞬移感。

Web 端輕量部署支援：提供 PyTorch 模型導出至 ONNX 格式腳本，未來可輕鬆整合至 Web (Three.js + MediaPipe) 實現真人視訊對打。

📁 專案結構 (Directory Structure)

.
├── models/                     # 訓練好的 PPO 模型權重 (.zip)
│   ├── self_play_model.zip      # 基礎控球 Baseline
│   └── fly_pingpong_attacker.zip# 微調後強攻模型
├── scene.xml                   # MuJoCo 3D 場景與飛行器物理定義
├── rally_env.py                # 自訂 Gymnasium 桌球環境邏輯
├── self_play_wrapper.py        # Self-Play 自我對弈封裝器
├── train_self_play.py          # 自我對弈訓練腳本
├── fine_tune_attack.py         # 溫和進攻意圖微調腳本
├── render_eval.py              # 視覺化即時渲染驗收腳本
├── export_onnx.py              # 匯出至 ONNX Web 模型腳本
└── README.md                   # 專案說明文件


🛠️ 環境需求與安裝 (Installation)

請確保已安裝 Python 3.10+，並透過以下命令安裝專案所需依賴套件：

pip install mujoco gymnasium stable-baselines3 torch onnx onnxruntime tensorboard


⚙️ 物理環境與參數設置 (Physics Configuration)

為了確保 AI 在具備揮拍爆發力的同時不會失控抖動，場景設定採納以下黃金比例：

物件 / 關節

物理參數設定

說明

機身質量

mass="0.01"

保持極佳的加速反應能力

滑軌關節 (X/Y/Z)

damping="0.08"

提供適度物理慣性阻力，消除高頻震盪

滑軌馬達 (Motor)

gear="7.0"

提供足夠滑行與強攻推力

拍面轉向馬達

gear="5.0"

確保拍面角度精準調節，引導斜線球

🚀 快速開始 (Quick Start)

1. 視覺化模型驗收 (Evaluation)

觀看已訓練好的 AI 進行現場實際對打與強攻展示：

python render_eval.py


2. 啟動 Self-Play 自我對弈訓練 (Training)

啟動多進程（Vectorized Env）Self-Play 訓練：

python train_self_play.py


你可以透過 TensorBoard 監控訓練指標：

tensorboard --logdir=./tb_logs


3. 微調強攻策略 (Fine-tuning)

若需引導現有 Baseline 模型嘗試更大力量與角度的抽球：

python fine_tune_attack.py


4. 匯出 ONNX 網頁端模型 (ONNX Export)

將訓練好的模型轉換為 ONNX 格式，供前端 Web Runtime 調用：

python export_onnx.py


📊 關鍵訓練指標指南 (Training Diagnostics)

在 TensorBoard 中監控以下指標，可確認模型是否健康收斂：

train/std：應穩定控制在 0.30 ~ 0.45。若飆升至 > 1.0 代表策略崩潰；若降至 < 0.1 代表過度僵硬。

train/explained_variance：應維持在 > 0.90，代表 Critic 網路精準預測回報。

rollout/ep_len_mean：穩定對打情況下應介於 150 ~ 500+（代表長來回對打）。

🛣️ 未來展望 (Roadmap)

[x] 完成 MuJoCo 物理動態調整與穩定 PPO Self-Play 訓練

[x] 完成強攻導向獎勵設計與 ONNX 模型轉換

[ ] 結合 Three.js 建立瀏覽器 3D 乒乓球渲染場景

[ ] 整合 MediaPipe Hands 手勢追蹤，實現 Webcam 真人 vs AI 即時對打

📜 授權協議 (License)

本專案採用 MIT License 授權。
