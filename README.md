# yolo-game-sb3

基于 **YOLO 目标检测 + PPO 强化学习**的魂斗罗（FC版）游戏 AI 训练框架。  
智能体仅通过屏幕画面（纯视觉输入）学习玩《魂斗罗》，无需访问游戏内存。

---

## 技术栈

| 组件 | 说明 |
|------|------|
| [stable-retro](https://github.com/Farama-Foundation/stable-retro) | NES 游戏模拟环境（Contra-Nes） |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | PPO 强化学习算法 |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | 实时目标检测（敌人、道具、危险区域） |
| OpenCV | 图像预处理、命数识别 |
| PyTorch | 神经网络后端 |
| Python ≥ 3.10 | 运行环境 |

---

## 项目结构

```
yolo-game-sb3/
├── contra_vision_env.py   # 自定义 Gym 环境（纯视觉观察 + YOLO 奖励）
├── train_contra.py        # PPO 训练脚本（含超参数配置）
├── test_visual.py         # 可视化测试：显示游戏画面与 YOLO 检测框
├── test_right.py          # 简单测试：固定向右移动动作验证环境
├── main.py                # 项目入口（占位）
├── pyproject.toml         # 依赖配置
├── models/
│   └── yolo-0327.pt       # 预训练 YOLO 检测模型
├── roms/
│   └── Contra (USA).nes   # 游戏 ROM 文件
├── assets/
│   └── lives_template.png # 命数识别模板（可选）
└── logs/
    ├── contra_ppo/        # 训练日志
    └── tensorboard/       # TensorBoard 日志
```

---

## 环境要求与安装

**Python 版本：** `>= 3.10`

```bash
# 安装依赖
pip install stable-baselines3>=2.0.0 ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0 torch>=2.0.0

# 安装 stable-retro（需要从源码安装以支持魂斗罗）
cd stable-retro/tmp/stable-retro
pip install -e .

# 导入游戏 ROM
python -c "import stable_retro; stable_retro.data.add_rom('roms/Contra (USA).nes')"
```

> 需要自行准备 `Contra (USA).nes` ROM 文件，并放置于 `roms/` 目录。

---

## 使用方法

### 训练

```bash
python train_contra.py
```

训练会自动：
- 启动 4 个并行环境（`SubprocVecEnv`）
- 每 50,000 步评估一次，保存最佳模型至 `models/contra_ppo/best_model/`
- 每 100,000 步保存检查点至 `models/contra_ppo/checkpoints/`

**手动中断**（Ctrl+C）后会自动保存当前模型。

### 测试已训练模型

```bash
python train_contra.py test <model_path> <vec_normalize_path>
# 例如：
python train_contra.py test models/contra_ppo/best_model/best_model.zip models/contra_ppo/vec_normalize.pkl
```

### 可视化测试（查看 YOLO 检测效果）

```bash
python test_visual.py
```

运行后弹出游戏窗口，显示实时 YOLO 检测框、奖励信息及对象计数。  
按 `q` 退出，`r` 重置，`空格` 暂停。

### 验证向右移动奖励

```bash
python test_right.py
```

使用固定向右动作运行 100 步，验证环境的向右奖励是否正常触发。

### 查看 TensorBoard

```bash
tensorboard --logdir logs/tensorboard/contra_ppo
```

---

## 奖励机制

奖励函数基于 YOLO 检测结果设计，通过**对象消失 + 像素变化**双重确认来判断击杀事件。

| 事件 | 奖励值 | 说明 |
|------|--------|------|
| 击杀普通敌人（mob） | +1.0 | 对象 IoU 消失 + 区域像素变化 |
| 击杀炮台（turret） | +1.5 | 同上 |
| 击中 Boss 弱点 | +5.0 | Boss 弱点区域消失 |
| 击杀 Boss | +10.0 | Boss 对象消失 |
| 躲避敌人子弹 | +0.3 | 子弹消失且玩家存活 |
| 拾取道具 | +2.0 | 道具对象消失 |
| **向右移动** | **+2.0** | 玩家 x 坐标增加 > 1px |
| 到达历史最远位置 | +0.5 | 额外进度奖励 |
| 原地不动 | -0.15 | 防止智能体停滞 |
| 向左移动 | -0.5 | 遏制回退行为 |
| 死亡（命数减少） | -1.0 | 每次死亡惩罚 |
| 进入水区域 | -0.3 | 危险区域警告 |
| 每步生存 | +0.01 | 存活鼓励 |

> 奖励值均在 `train_contra.py` 的 `REWARD_CONFIG` 中集中配置，可按训练效果自由调整。

---

## 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `N_ENVS` | 4 | 并行环境数量 |
| `FRAME_STACK` | 4 | 帧堆叠数（观察形状 `(4, 84, 84)`） |
| `LEARNING_RATE` | 2.5e-4 | PPO 学习率 |
| `N_STEPS` | 256 | 每次收集的时间步数 |
| `BATCH_SIZE` | 256 | 每次更新的 mini-batch 大小 |
| `N_EPOCHS` | 4 | 每批数据的更新轮数 |
| `ENT_COEF` | 0.005 | 熵系数（控制探索程度） |
| `GAMMA` | 0.98 | 折扣因子 |
| `CLIP_RANGE` | 0.2 | PPO 裁剪范围 |
| `TOTAL_TIMESTEPS` | 10,000,000 | 总训练步数 |
| `EVAL_FREQ` | 50,000 | 评估频率（步） |
| `CHECKPOINT_FREQ` | 100,000 | 检查点保存频率（步） |
