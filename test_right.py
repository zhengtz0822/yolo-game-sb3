"""
简单测试脚本 - 使用固定动作让玩家向右走
"""

import os
import time
import cv2
import numpy as np
from contra_vision_env import ContraVisionEnv, RewardConfig

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# 颜色配置
COLORS = {
    "player": (255, 255, 255),
    "mob": (0, 0, 255),
    "turret": (0, 165, 255),
    "ebullet": (0, 255, 255),
    "pit": (255, 0, 0),
    "water": (255, 0, 255),
    "item": (0, 255, 0),
    "boss-1": (0, 0, 128),
    "boss-1_weakness": (255, 255, 0),
}

# 动作映射 (Contra-Nes Discrete 36)
# Action 2: RIGHT (向右走)
# Action 3: RIGHT + A (向右射击)
# Action 19: UP + RIGHT + A (右上射击)
# Action 10: DOWN + RIGHT (向右蹲下)
# Action 11: DOWN + RIGHT + A (向右蹲下射击)

RIGHT_ACTIONS = [2, 3, 10, 11, 18, 19]  # 向右移动的动作

def main():
    print("=" * 60)
    print("测试向右移动策略")
    print("=" * 60)

    env = ContraVisionEnv(
        yolo_model_path='models/yolo-0327.pt',
        frame_stack=1,
    )

    print(f"\n向右动作: {RIGHT_ACTIONS}")
    print("这些动作包含 RIGHT 按钮\n")

    obs, info = env.reset()

    total_reward = 0
    right_count = 0

    # 运行 100 步，始终选择向右移动
    for i in range(100):
        # 选择向右移动的动作
        action = RIGHT_ACTIONS[i % len(RIGHT_ACTIONS)]

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        reward_details = info.get('reward_details', {})
        if reward_details.get('move_right', 0) > 0:
            right_count += 1

        if i % 20 == 0:
            player_x = env.prev_player_x
            print(f"Step {i}: X={player_x:.0f}, reward={total_reward:.2f}")

        if done or truncated:
            print(f"\n游戏结束于 Step {i}")
            break

    print(f"\n结果:")
    print(f"  向右移动次数: {right_count}")
    print(f"  最终 X 坐标: {env.prev_player_x:.0f}")
    print(f"  总奖励: {total_reward:.2f}")

    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    main()