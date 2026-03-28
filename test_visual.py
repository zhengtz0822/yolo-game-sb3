"""
测试脚本 - 显示 Contra 游戏画面
运行此脚本可以看到实际的游戏画面和 YOLO 检测结果
"""

import os
import time
import cv2
import numpy as np
from contra_vision_env import ContraVisionEnv, RewardConfig

# 配置
YOLO_MODEL_PATH = "models/yolo-0327.pt"
DISPLAY_SCALE = 2  # 画面放大倍数

# 颜色配置 (BGR)
COLORS = {
    "player": (255, 255, 255),  # 白色
    "mob": (0, 0, 255),        # 红色
    "turret": (0, 165, 255),   # 橙色
    "ebullet": (0, 255, 255),  # 黄色
    "pit": (255, 0, 0),        # 蓝色
    "water": (255, 0, 255),    # 紫色
    "bridge": (128, 128, 128), # 灰色
    "item": (0, 255, 0),       # 绿色
    "boss-1": (0, 0, 128),     # 深红
    "boss-1_weakness": (255, 255, 0),  # 青色
}

def draw_detections(frame, objects, class_ids):
    """在画面上绘制检测结果"""
    total_objects = 0
    for class_name, boxes in objects.items():
        color = COLORS.get(class_name, (255, 255, 255))

        for box in boxes:
            x1, y1, x2, y2 = box
            # 绘制边界框（更粗的线条）
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            # 绘制标签（更大字体）
            label = class_name
            cv2.putText(frame, label, (int(x1), int(y1) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            total_objects += 1

    return frame, total_objects

def get_player_x(objects):
    """获取玩家 x 坐标"""
    player_boxes = objects.get("player", [])
    if player_boxes:
        box = player_boxes[0]
        return (box[0] + box[2]) / 2
    return 0

def main():
    print("=" * 60)
    print("Contra 视觉环境测试")
    print("=" * 60)

    # 创建奖励配置
    reward_config = RewardConfig(
        kill_mob=1.0,
        kill_turret=1.5,
        hit_boss_weakness=5.0,
        kill_boss=10.0,
        dodge_bullet=0.3,
        pickup_item=2.0,
    )

    # 创建环境
    print("\n创建环境...")
    env = ContraVisionEnv(
        yolo_model_path=YOLO_MODEL_PATH,
        reward_config=reward_config,
        frame_stack=1,
    )

    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space} ({env.action_space.n} 个动作)")

    print("\n按 'q' 退出, 'r' 重置, 空格暂停")
    print("方向键控制移动, 'z' 射击, 'x' 跳跃\n")

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

    # 简单的动作映射
    # Discrete(36) 动作空间
    # 0: 无操作
    # 其他动作需要根据实际游戏测试

    while running:
        start_time = time.time()

        if not paused:
            # 随机动作（演示用）
            action = env.action_space.sample()

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # 获取原始画面（从 retro 环境）
            raw_frame = env.prev_frame_raw.copy()

            # 绘制检测结果
            total_detected = 0
            if hasattr(env, 'prev_objects'):
                raw_frame, total_detected = draw_detections(raw_frame, env.prev_objects, env.class_ids)

            # 获取玩家 x 坐标
            player_x = get_player_x(env.prev_objects) if hasattr(env, 'prev_objects') else 0

            # 显示信息
            info_text = [
                f"Step: {step_count}",
                f"Reward: {total_reward:.2f}",
                f"Lives: {info.get('lives', 'N/A')}",
                f"Kills: {info.get('total_kills', 0)}",
                f"Player X: {player_x:.0f}",
                f"Detected: {total_detected}",
            ]

            for i, text in enumerate(info_text):
                cv2.putText(raw_frame, text, (10, 20 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示检测到的对象数量
            object_counts = info.get('object_counts', {})
            if object_counts:
                y_offset = 160
                count_text = []
                for name, count in object_counts.items():
                    if count > 0:
                        count_text.append(f"{name}:{count}")
                if count_text:
                    cv2.putText(raw_frame, " | ".join(count_text[:5]), (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 显示奖励详情
            reward_details = info.get('reward_details', {})
            if reward_details:
                y_offset = 180
                for key, value in reward_details.items():
                    if value != 0:
                        text = f"{key}: {value:+.2f}"
                        color = (0, 255, 0) if value > 0 else (0, 0, 255)
                        cv2.putText(raw_frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        y_offset += 15

            # 放大画面
            display_frame = cv2.resize(raw_frame,
                                       (raw_width * DISPLAY_SCALE, raw_height * DISPLAY_SCALE))

            # 显示画面
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