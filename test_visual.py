"""
测试脚本 - 显示 Contra 游戏画面
运行此脚本可以看到实际的游戏画面和 RAM 变量信息
支持加载训练好的模型进行推理，或使用随机动作演示
"""

import os
import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from contra_vision_env import ContraEnv, RewardConfig

# 配置
DISPLAY_SCALE = 2  # 画面放大倍数

# 模型路径配置
MODEL_PATH = "models/contra_ppo/best_model/best_model.zip"  # 优先加载最佳模型
FALLBACK_MODEL_PATH = "models/contra_ppo/final_model.zip"  # 备用：最终模型


def load_model():
    """尝试加载训练好的模型，失败则返回 None"""
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print(f"已加载模型: {MODEL_PATH}")
        return model
    elif FALLBACK_MODEL_PATH and os.path.exists(FALLBACK_MODEL_PATH):
        model = PPO.load(FALLBACK_MODEL_PATH)
        print(f"已加载备用模型: {FALLBACK_MODEL_PATH}")
        return model
    else:
        print("未找到模型文件，使用随机动作")
        return None


def main():
    print("=" * 60)
    print("Contra 视觉环境测试")
    print("=" * 60)

    # 尝试加载模型
    model = load_model()
    use_model = model is not None
    mode_text = "模型推理" if use_model else "随机动作"

    # 创建奖励配置
    reward_config = RewardConfig(
        progress_coef=1.0,
        progress_penalty=-0.3,
        no_progress_penalty=-0.1,
        new_max_bonus=0.5,
        score_coef=0.1,
        death_penalty=-2.0,
        survival_bonus=0.005,
    )

    # 创建环境（frame_stack=4 与训练保持一致）
    print("\n创建环境...")
    env = ContraEnv(frame_stack=4, reward_config=reward_config, render_mode="human")

    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space} ({env.action_space.n} 个动作)")

    print(f"\n按 'q' 退出, 'r' 重置, 空格暂停")
    print(f"当前模式: {mode_text}\n")

    # 重置环境
    obs, info = env.reset()

    # 获取原始画面尺寸
    raw_height, raw_width = 224, 240  # NES 分辨率

    running = True
    paused = False
    step_count = 0
    total_reward = 0
    fps = 30
    frame_time = 1.0 / fps

    while running:
        start_time = time.time()

        if not paused:
            # 动作选择：有模型则推理，否则随机
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # 获取原始画面
            raw_frame = env.prev_frame_raw.copy()

            # 读取 RAM 变量
            xscroll = info.get('xscroll', 0)
            score = info.get('score', 0)
            lives = info.get('lives', 'N/A')
            player_state = info.get('player_state', 0)

            # 在左上角显示 RAM 变量信息
            ram_info = [
                f"Step: {step_count}",
                f"Total Reward: {total_reward:.2f}",
                f"Step Reward: {reward:+.3f}",
                f"xscroll: {xscroll}",
                f"score: {score}",
                f"lives: {lives}",
                f"player_state: {player_state}",
                f"max_xscroll: {info.get('max_xscroll', 0)}",
            ]

            for i, text in enumerate(ram_info):
                cv2.putText(raw_frame, text, (5, 14 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

            # 显示奖励详情
            reward_details = info.get('reward_details', {})
            if reward_details:
                y_offset = 14 + len(ram_info) * 16 + 4
                cv2.putText(raw_frame, "Reward Details:", (5, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                y_offset += 14
                for key, value in reward_details.items():
                    if value != 0:
                        text = f"  {key}: {value:+.3f}"
                        color = (0, 255, 0) if value > 0 else (0, 0, 255)
                        cv2.putText(raw_frame, text, (5, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                        y_offset += 13

            # 在右上角显示当前模式
            mode_color = (0, 255, 128) if use_model else (255, 128, 0)
            cv2.putText(raw_frame, mode_text, (raw_width - 80, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1)

            # 放大画面
            display_frame = cv2.resize(raw_frame,
                                       (raw_width * DISPLAY_SCALE, raw_height * DISPLAY_SCALE))

            # 显示画面（retro 返回 RGB，需转 BGR 给 OpenCV）
            cv2.imshow("Contra - Press 'q' to quit", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

            if done or truncated:
                print(f"\n游戏结束! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
                obs, info = env.reset()
                step_count = 0
                total_reward = 0

        # 处理按键
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord('r'):
            obs, info = env.reset()
            step_count = 0
            total_reward = 0
            print("环境已重置")
        elif key == ord(' '):
            paused = not paused
            print("暂停" if paused else "继续")

        # 控制帧率
        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    # 清理
    env.close()
    cv2.destroyAllWindows()
    print("\n测试结束")


if __name__ == "__main__":
    main()
