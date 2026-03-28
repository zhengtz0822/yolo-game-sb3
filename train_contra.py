"""
魂斗罗强化学习训练脚本
使用 PPO 算法训练一个基于纯视觉的智能体玩 FC 版《魂斗罗》
"""

import os
import time
from typing import Callable

import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from contra_vision_env import ContraVisionEnv, RewardConfig

# ==================== 配置参数 ====================

# YOLO 模型路径（预训练的敌人检测模型）
YOLO_MODEL_PATH = "models/yolo-0327.pt"

# 命数模板图片路径（可选，用于模板匹配法识别命数）
LIVES_TEMPLATE_PATH = "assets/lives_template.png"  # 设为 None 则使用像素统计法

# 命数识别区域 (x, y, w, h) - 左上角区域
LIVES_ROI = (10, 20, 50, 40)

# 游戏状态检测区域
GAME_OVER_ROI = (100, 100, 120, 30)

# 环境参数
FRAME_STACK = 4  # 帧堆叠数量

# ==================== 奖励配置 ====================
# 可根据训练效果调整各项奖励值
REWARD_CONFIG = RewardConfig(
    # 敌人击杀奖励
    kill_mob=1.0,          # 击杀普通敌人
    kill_turret=1.5,       # 击杀炮台
    hit_boss_weakness=5.0, # 击中Boss弱点
    kill_boss=10.0,        # 击杀Boss

    # 躲避/受伤奖励
    dodge_bullet=0.3,      # 成功躲避子弹
    hit_by_bullet=-0.5,    # 被子弹击中

    # 危险区域奖励
    fall_into_pit=-1.0,    # 掉入坑洞
    enter_water=-0.3,      # 进入水区域

    # 道具奖励
    pickup_item=2.0,       # 拾取道具

    # 命数变化奖励
    lives_decrease=-1.0,   # 命数减少（死亡）
    lives_increase=0.5,    # 命数增加（加命）
    survival_bonus=0.01,   # 生存鼓励（每步）↑ 从0.001提升，加强存活激励

    # 进度奖励
    move_right_bonus=2.0,   # 向右移动奖励 ↑ 从0.5提升至2.0，使其与击杀敌人奖励相当，强制驱动前进
    move_left_penalty=-0.5, # 向左移动惩罚 ↑ 从-0.2加强至-0.5，遏制向左回退行为
    no_move_penalty=-0.15,  # 原地不动惩罚 ↑ 从-0.01提高15倍，杜绝原地徘徊

    # 检测阈值
    iou_threshold=0.3,           # IoU阈值（判断对象消失）
    pixel_change_threshold=30.0, # 像素变化阈值（确认击杀）
)

# 并行环境数量
N_ENVS = 4

# PPO 超参数
LEARNING_RATE = 2.5e-4
N_STEPS = 256        # ↑ 从128增加至256，收集更长经验序列，减少梯度噪声
BATCH_SIZE = 256     # 保持256，N_STEPS*N_ENVS=1024，256可整除
N_EPOCHS = 4
ENT_COEF = 0.005     # ↓ 从0.01降低，减少过度随机探索，让策略更稳定地向右前进
GAMMA = 0.98         # ↓ 从0.99降低，更重视近期奖励，加快对向右移动信号的响应
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# ==================== 调试/正式训练模式 ====================
DEBUG_MODE = False  # True: 本地调试模式（可视化、快速验证）  False: 正式训练模式（高性能、长时间）

# 训练参数（根据模式自动调整）
TOTAL_TIMESTEPS = 500_000 if DEBUG_MODE else 10_000_000   # 调试: 50万步  正式: 1000万步
RENDER_ENABLED = DEBUG_MODE  # 是否启用实时渲染回调
RENDER_FREQ = 2000           # 渲染回调频率（仅 DEBUG_MODE 生效）
RENDER_STEPS = 300           # 每次渲染步数（仅 DEBUG_MODE 生效）

# 保存路径
LOG_DIR = "logs/contra_ppo"
MODEL_SAVE_DIR = "models/contra_ppo"
TENSORBOARD_LOG = "logs/tensorboard/contra_ppo"

# 回调参数
EVAL_FREQ = 10_000 if DEBUG_MODE else 100_000  # 评估频率  调试: 1万步  正式: 10万步
N_EVAL_EPISODES = 3 if DEBUG_MODE else 2  # 每次评估的回合数
CHECKPOINT_FREQ = 100_000  # 检查点保存频率

# ==================== 环境工厂函数 ====================


def make_env(
    yolo_model_path: str,
    lives_template_path: str = None,
    reward_config: RewardConfig = None,
    frame_stack: int = 1,  # 设为1，使用 VecFrameStack 进行堆叠
    lives_roi: tuple = (10, 20, 50, 40),
    game_over_roi: tuple = (100, 100, 120, 30),
    rank: int = 0,
    seed: int = 0,
    render_mode: str = "rgb_array",
) -> Callable:
    """
    创建环境的工厂函数

    Args:
        yolo_model_path: YOLO 模型路径
        lives_template_path: 命数模板路径
        reward_config: 奖励配置
        frame_stack: 帧堆叠数（内部使用）
        lives_roi: 命数识别区域
        game_over_roi: 游戏结束检测区域
        rank: 环境序号（用于多进程）
        seed: 随机种子
        render_mode: 渲染模式，"rgb_array" 或 "human"

    Returns:
        环境创建函数
    """

    def _init():
        # 创建 ContraVisionEnv 实例
        env = ContraVisionEnv(
            yolo_model_path=yolo_model_path,
            lives_template_path=lives_template_path,
            reward_config=reward_config,
            frame_stack=frame_stack,  # 内部不堆叠，由 VecFrameStack 处理
            lives_roi=lives_roi,
            game_over_roi=game_over_roi,
            render_mode=render_mode,
        )

        # 包装 Monitor 以记录统计信息
        env = Monitor(env)

        # 设置随机种子
        env.reset(seed=seed + rank)

        return env

    return _init


# ==================== 实时渲染回调 ====================


class RenderCallback(BaseCallback):
    """训练过程中定期渲染游戏画面的回调"""

    def __init__(self, render_env, render_freq=5000, n_render_steps=200, verbose=0):
        super().__init__(verbose)
        self.render_env = render_env
        self.render_freq = render_freq
        self.n_render_steps = n_render_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            obs = self.render_env.reset()
            for _ in range(self.n_render_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.render_env.step(action)
                self.render_env.render()
                time.sleep(1 / 30)  # ~30fps
                if done.any():
                    obs = self.render_env.reset()
        return True


# ==================== 主训练函数 ====================


def main():
    """主训练函数"""
    print("=" * 60)
    print("魂斗罗强化学习训练")
    print("=" * 60)

    # 检查 YOLO 模型是否存在
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"错误: YOLO 模型文件不存在: {YOLO_MODEL_PATH}")
        print("请确保已训练好 YOLO 模型并指定正确路径")
        return

    # 检查命数模板（如果指定）
    if LIVES_TEMPLATE_PATH and not os.path.exists(LIVES_TEMPLATE_PATH):
        print(f"警告: 命数模板文件不存在: {LIVES_TEMPLATE_PATH}")
        print("将使用像素统计法识别命数")

    # 创建保存目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)

    print(f"\n配置信息:")
    print(f"  - 训练模式: {'🔧 调试模式' if DEBUG_MODE else '🚀 正式训练'}")
    print(f"  - YOLO 模型: {YOLO_MODEL_PATH}")
    print(f"  - 命数模板: {LIVES_TEMPLATE_PATH}")
    print(f"  - 并行环境数: {N_ENVS}")
    print(f"  - 总训练步数: {TOTAL_TIMESTEPS:,}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 帧堆叠数: {FRAME_STACK}")
    print(f"\n奖励配置:")
    print(f"  - 击杀敌人(mob): {REWARD_CONFIG.kill_mob}")
    print(f"  - 击杀炮台(turret): {REWARD_CONFIG.kill_turret}")
    print(f"  - 击中Boss弱点: {REWARD_CONFIG.hit_boss_weakness}")
    print(f"  - 击杀Boss: {REWARD_CONFIG.kill_boss}")
    print(f"  - 躲避子弹: {REWARD_CONFIG.dodge_bullet}")
    print(f"  - 拾取道具: {REWARD_CONFIG.pickup_item}")

    # 创建并行环境
    print("\n创建并行环境...")
    env = SubprocVecEnv(
        [
            make_env(
                yolo_model_path=YOLO_MODEL_PATH,
                lives_template_path=LIVES_TEMPLATE_PATH,
                reward_config=REWARD_CONFIG,
                frame_stack=1,  # 环境内部不堆叠
                lives_roi=LIVES_ROI,
                game_over_roi=GAME_OVER_ROI,
                rank=i,
                seed=42,
            )
            for i in range(N_ENVS)
        ]
    )

    # 使用 VecFrameStack 堆叠帧
    # 观察空间将变为 (4, 84, 84)
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")

    # 使用 VecNormalize 对观察和奖励进行归一化
    # 这有助于训练稳定性
    env = VecNormalize(
        env,
        norm_obs=False,  # 图像观察不做归一化（CnnPolicy 要求 uint8 图像）
        norm_reward=True,  # 归一化奖励
        clip_obs=10.0,  # 观察裁剪范围
        clip_reward=10.0,  # 奖励裁剪范围
        gamma=GAMMA,  # 折扣因子
    )

    print(f"环境观察空间: {env.observation_space}")
    print(f"环境动作空间: {env.action_space}")

    # 创建 PPO 模型
    print("\n创建 PPO 模型...")
    model = PPO(
        "CnnPolicy",  # 使用 CNN 策略（适合图像输入）
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG,
        device="cuda",  # 自动选择设备（优先使用 GPU）
    )

    print(f"模型架构:")
    print(model.policy)

    # 创建回调函数
    callbacks = []

    # 1. 评估回调 - 定期评估并保存最佳模型
    eval_env = DummyVecEnv(
        [
            make_env(
                yolo_model_path=YOLO_MODEL_PATH,
                lives_template_path=LIVES_TEMPLATE_PATH,
                reward_config=REWARD_CONFIG,
                frame_stack=1,
                lives_roi=LIVES_ROI,
                game_over_roi=GAME_OVER_ROI,
                rank=0,
                seed=100,
                render_mode="human" if DEBUG_MODE else "rgb_array",
            )
        ]
    )
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="first")
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,  # 图像观察不做归一化
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=DEBUG_MODE,  # 调试模式下评估时渲染画面
        verbose=1,
    )
    callbacks.append(eval_callback)

    # 2. 检查点回调 - 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(MODEL_SAVE_DIR, "checkpoints"),
        name_prefix="contra_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # 3. 实时渲染回调（复用 eval_env，避免同进程多模拟器实例冲突）
    if RENDER_ENABLED:
        render_callback = RenderCallback(eval_env, render_freq=RENDER_FREQ, n_render_steps=RENDER_STEPS)
        callbacks.append(render_callback)

    # 合并回调
    callback = CallbackList(callbacks)

    # 开始训练
    print("\n开始训练...")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")

    # 保存最终模型
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model")
    model.save(final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")

    # 保存 VecNormalize 统计信息
    vec_normalize_path = os.path.join(MODEL_SAVE_DIR, "vec_normalize.pkl")
    env.save(vec_normalize_path)
    print(f"VecNormalize 统计信息已保存至: {vec_normalize_path}")

    # 关闭环境
    env.close()
    eval_env.close()

    print("\n训练完成!")
    print("=" * 60)


def load_and_test(model_path: str, vec_normalize_path: str):
    """
    加载模型并测试

    Args:
        model_path: 模型路径
        vec_normalize_path: VecNormalize 统计文件路径
    """
    print(f"加载模型: {model_path}")

    # 创建测试环境
    env = ContraVisionEnv(
        yolo_model_path=YOLO_MODEL_PATH,
        lives_template_path=LIVES_TEMPLATE_PATH,
        reward_config=REWARD_CONFIG,
        frame_stack=1,
        lives_roi=LIVES_ROI,
        game_over_roi=GAME_OVER_ROI,
    )
    env = Monitor(env)
    env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="first")
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # 测试模式下不更新统计
    env.norm_reward = False  # 测试时不归一化奖励

    # 加载模型
    model = PPO.load(model_path, env=env)

    # 运行测试
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("\n开始测试...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 100 == 0:
            print(f"Step {step}, Reward: {total_reward:.2f}")

    print(f"\n测试结束:")
    print(f"  - 总步数: {step}")
    print(f"  - 总奖励: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式
        if len(sys.argv) < 4:
            print("用法: python train_contra.py test <model_path> <vec_normalize_path>")
            sys.exit(1)
        load_and_test(sys.argv[2], sys.argv[3])
    else:
        # 训练模式
        main()