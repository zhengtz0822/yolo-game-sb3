"""
魂斗罗视觉强化学习环境
基于纯视觉输入（屏幕画面）的 Gym 环境，用于训练强化学习智能体玩 FC 版《魂斗罗》
"""

import stable_retro as retro
import gymnasium as gym
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set
from ultralytics import YOLO


@dataclass
class RewardConfig:
    """
    奖励配置类

    定义各类检测对象对应的奖励值
    """

    # === 敌人击杀奖励 ===
    # 击杀敌人 (mob)
    kill_mob: float = 1.0
    # 击杀炮台 (turret)
    kill_turret: float = 1.5
    # 击中 boss 弱点 (boss-1_weakness)
    hit_boss_weakness: float = 5.0
    # 击杀 boss (boss-1)
    kill_boss: float = 10.0

    # === 躲避/受伤奖励 ===
    # 成功躲避敌人子弹 (ebullet 消失且玩家存活)
    dodge_bullet: float = 0.3
    # 被敌人子弹击中
    hit_by_bullet: float = -0.5

    # === 危险区域奖励 ===
    # 掉入坑洞 (pit)
    fall_into_pit: float = -1.0
    # 进入水区域 (water) - 可能危险
    enter_water: float = -0.3

    # === 道具奖励 ===
    # 拾取道具 (item)
    pickup_item: float = 2.0

    # === 命数变化奖励 ===
    # 命数减少（死亡）
    lives_decrease: float = -1.0
    # 命数增加（加命）
    lives_increase: float = 0.5
    # 生存鼓励（每步）
    survival_bonus: float = 0.001

    # === 进度奖励 ===
    # 向右移动奖励（鼓励探索）
    move_right_bonus: float = 0.5
    # 向左移动惩罚（防止来回移动）
    move_left_penalty: float = -0.2
    # 原地不动惩罚（防止发呆）
    no_move_penalty: float = -0.01

    # === 检测阈值 ===
    # IoU 阈值，用于判断对象消失
    iou_threshold: float = 0.3
    # 像素变化阈值，用于确认击杀
    pixel_change_threshold: float = 30.0


class ContraVisionEnv(gym.Env):
    """
    基于 pure vision 的魂斗罗强化学习环境

    观察空间：堆叠4帧后的灰度图，形状 (4, 84, 84)，像素值 0-255
    动作空间：使用 retro 的离散动作空间
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # YOLO 类别名称定义
    CLASS_NAMES = {
        "player": "玩家",
        "mob": "普通敌人",
        "turret": "炮台",
        "ebullet": "敌人子弹",
        "pit": "坑洞",
        "water": "水区域",
        "bridge": "桥",
        "item": "道具",
        "boss-1": "Boss",
        "boss-1_weakness": "Boss弱点",
    }

    def __init__(
        self,
        yolo_model_path: str,
        lives_template_path: Optional[str] = None,
        frame_stack: int = 4,
        reward_config: Optional[RewardConfig] = None,
        # 命数识别区域参数（左上角区域）
        lives_roi: Tuple[int, int, int, int] = (10, 20, 50, 40),  # (x, y, w, h)
        # 像素统计法的命数阈值（需要根据实际标定）
        lives_pixel_thresholds: Optional[List[float]] = None,
        # 游戏状态区域（用于判断游戏是否结束）
        game_over_roi: Tuple[int, int, int, int] = (100, 100, 120, 30),
        # 渲染模式
        render_mode: str = "rgb_array",
    ):
        """
        初始化环境

        Args:
            yolo_model_path: 预训练的 YOLO 模型路径
            lives_template_path: 可选，左上角命数模板图片路径（灰度图）
            frame_stack: 堆叠帧数，默认 4
            reward_config: 奖励配置，若为 None 则使用默认配置
            lives_roi: 命数识别区域 (x, y, w, h)
            lives_pixel_thresholds: 像素统计法的命数阈值列表
            game_over_roi: 游戏结束检测区域
        """
        super().__init__()

        # 创建 retro 环境（Contra - Level1）
        # 使用 RetroEnv 直接创建，支持 experimental 游戏
        self.retro_env = retro.RetroEnv(
            game="Contra-Nes",
            state="Level1",
            inttype=retro.data.Integrations.ALL,  # 包含 experimental 游戏
            use_restricted_actions=retro.Actions.DISCRETE,  # 使用离散动作空间
            render_mode=render_mode,  # 渲染模式，"human" 直接显示窗口，"rgb_array" 返回数组
        )

        # 复制 retro 环境的观察和动作空间
        self.action_space = self.retro_env.action_space

        # 定义观察空间：堆叠 frame_stack 帧的 84x84 灰度图
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, 84, 84),
            dtype=np.uint8,
        )

        # 加载 YOLO 模型并获取各类别 ID
        self.yolo_model = YOLO(yolo_model_path)
        self.class_ids = self._get_class_ids()

        # 打印类别信息
        print(f"YOLO 模型类别映射:")
        for name, desc in self.CLASS_NAMES.items():
            class_id = self.class_ids.get(name, -1)
            if class_id >= 0:
                print(f"  - {name} ({desc}): ID={class_id}")

        # 命数模板（用于模板匹配）
        self.lives_template = None
        if lives_template_path is not None:
            self.lives_template = cv2.imread(lives_template_path, cv2.IMREAD_GRAYSCALE)
            if self.lives_template is None:
                print(f"警告: 无法加载命数模板图片: {lives_template_path}，将使用像素统计法识别命数")

        # 奖励配置
        self.reward_config = reward_config or RewardConfig()

        # 可配置参数
        self.lives_roi = lives_roi
        self.lives_pixel_thresholds = lives_pixel_thresholds or [0.1, 0.2, 0.3, 0.4]
        self.game_over_roi = game_over_roi

        # 状态变量
        self._reset_state()

        # 原始画面尺寸（从 retro 环境获取）
        self.raw_height, self.raw_width = self.retro_env.observation_space.shape[:2]

    def _get_class_ids(self) -> Dict[str, int]:
        """
        从 YOLO 模型中获取所有预定义类别的 ID

        Returns:
            类别名称到 ID 的映射字典
        """
        class_ids = {}
        model_class_names = self.yolo_model.names

        # 查找每个预定义类别对应的 ID
        for class_name in self.CLASS_NAMES.keys():
            for model_id, model_name in model_class_names.items():
                if model_name == class_name:
                    class_ids[class_name] = model_id
                    break

        # 检查是否有未找到的类别
        missing_classes = [
            name for name in self.CLASS_NAMES.keys() if name not in class_ids
        ]
        if missing_classes:
            print(
                f"警告: YOLO 模型中未找到以下类别: {missing_classes}, "
                f"可用类别: {list(model_class_names.values())}"
            )

        return class_ids

    def _reset_state(self):
        """重置所有状态变量"""
        self.frame_buffer = np.zeros(
            (self.frame_stack, 84, 84), dtype=np.uint8
        )  # 帧堆叠缓冲区
        self.prev_frame_raw = None  # 上一帧原始画面

        # 各类对象的上一帧检测结果（按类别存储）
        self.prev_objects: Dict[str, List[Tuple[int, int, int, int]]] = {
            "player": [],
            "mob": [],
            "turret": [],
            "ebullet": [],
            "pit": [],
            "water": [],
            "bridge": [],
            "item": [],
            "boss-1": [],
            "boss-1_weakness": [],
        }

        self.prev_action = 0  # 上一动作
        self.prev_lives = 3  # 上一命数（初始为 3）
        self.total_kills = 0  # 累计击杀数
        self.prev_player_x = 0  # 上一帧玩家 x 坐标
        self.max_player_x = 0  # 最大 x 坐标（用于防止刷分）
        self.current_step = 0  # 当前 episode 步数
        self.max_episode_steps = 4000  # 单 episode 最大步数，防止评估卡死
        self.detect_interval = 3  # 每 N 帧检测一次 YOLO，中间帧复用上次结果

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理单帧画面

        Args:
            frame: 原始 BGR 画面

        Returns:
            84x84 的灰度图
        """
        # BGR 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 缩放至 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        return resized

    def _stack_frames(self, new_frame: np.ndarray) -> np.ndarray:
        """
        更新帧堆叠缓冲区

        Args:
            new_frame: 新的预处理后帧

        Returns:
            堆叠后的观察
        """
        # 将帧缓冲区向前滚动，新帧放在最后
        self.frame_buffer[:-1] = self.frame_buffer[1:]
        self.frame_buffer[-1] = new_frame

        return self.frame_buffer.copy()

    def _detect_objects(
        self, frame: np.ndarray, conf_threshold: float = 0.5
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        使用 YOLO 检测所有预定义类别的对象

        Args:
            frame: 原始 BGR 画面
            conf_threshold: 置信度阈值

        Returns:
            类别名称到边界框列表的映射字典
            {"mob": [(x1,y1,x2,y2), ...], "turret": [...], ...}
        """
        objects: Dict[str, List[Tuple[int, int, int, int]]] = {
            name: [] for name in self.CLASS_NAMES.keys()
        }

        # YOLO 推理
        results = self.yolo_model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # 获取类别和置信度
                    cls_id = int(boxes.cls[i].item())
                    conf = boxes.conf[i].item()

                    if conf > conf_threshold:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                        # 根据类别 ID 找到对应的类别名称
                        for name, id_ in self.class_ids.items():
                            if id_ == cls_id:
                                objects[name].append((x1, y1, x2, y2))
                                break

        return objects

    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """
        计算两个边界框的 IoU（交并比）

        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)

        Returns:
            IoU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _check_object_disappeared(
        self,
        prev_box: Tuple[int, int, int, int],
        curr_boxes: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.3,
    ) -> bool:
        """
        检查某个对象是否消失（通过 IoU 判断）

        Args:
            prev_box: 上一帧的对象框
            curr_boxes: 当前帧的同类别对象框列表
            iou_threshold: IoU 阈值

        Returns:
            是否消失
        """
        max_iou = 0.0
        for curr_box in curr_boxes:
            iou = self._compute_iou(prev_box, curr_box)
            max_iou = max(max_iou, iou)

        return max_iou < iou_threshold

    def _check_pixel_change(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        box: Tuple[int, int, int, int],
        threshold: float = 30.0,
    ) -> bool:
        """
        检查某个区域的像素变化是否超过阈值

        Args:
            prev_frame: 上一帧原始画面
            curr_frame: 当前帧原始画面
            box: 检测区域边界框
            threshold: 像素变化阈值

        Returns:
            是否有显著变化
        """
        x1, y1, x2, y2 = box

        # 确保坐标在画面范围内
        x1 = max(0, min(x1, self.raw_width - 1))
        x2 = max(0, min(x2, self.raw_width - 1))
        y1 = max(0, min(y1, self.raw_height - 1))
        y2 = max(0, min(y2, self.raw_height - 1))

        if x2 <= x1 or y2 <= y1:
            return False

        # 提取区域
        prev_region = prev_frame[y1:y2, x1:x2]
        curr_region = curr_frame[y1:y2, x1:x2]

        # 转为灰度图
        prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_region, cv2.COLOR_BGR2GRAY)

        # 计算像素变化均值
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_change = np.mean(diff)

        return mean_change > threshold

    def _compute_rewards_from_objects(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_objects: Dict[str, List[Tuple[int, int, int, int]]],
        curr_objects: Dict[str, List[Tuple[int, int, int, int]]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        根据对象检测结果计算奖励

        Args:
            prev_frame: 上一帧原始画面
            curr_frame: 当前帧原始画面
            prev_objects: 上一帧所有类别对象检测结果
            curr_objects: 当前帧所有类别对象检测结果

        Returns:
            (总奖励, 奖励详情字典)
        """
        reward = 0.0
        reward_details = {
            "kill_mob": 0,
            "kill_turret": 0,
            "hit_boss_weakness": 0,
            "kill_boss": 0,
            "dodge_bullet": 0,
            "hit_by_bullet": 0,
            "fall_into_pit": 0,
            "enter_water": 0,
            "pickup_item": 0,
            "move_right": 0,
            "move_left": 0,
            "no_move": 0,
        }

        config = self.reward_config

        # === 玩家位置跟踪和向右移动奖励 ===
        curr_player_boxes = curr_objects.get("player", [])
        if curr_player_boxes:
            # 获取玩家中心 x 坐标
            player_box = curr_player_boxes[0]  # 取第一个玩家
            curr_player_x = (player_box[0] + player_box[2]) / 2

            # 计算移动距离
            x_movement = curr_player_x - self.prev_player_x

            if x_movement > 1:  # 向右移动（阈值从>2降至>1，对小幅前进更敏感）
                # 直接给固定奖励，再加距离奖励
                reward += config.move_right_bonus
                reward_details["move_right"] = 1
                # 更新最大 x 坐标，并给予到达新位置的额外进度奖励
                if curr_player_x > self.max_player_x:
                    # 到达历史最远位置，额外奖励0.5，鼓励持续探索新区域
                    reward += 0.5
                    self.max_player_x = curr_player_x

            elif x_movement < -1:  # 向左移动（阈值从<-2调整为<-1，与右移阈值对称）
                reward += config.move_left_penalty
                reward_details["move_left"] = 1

            else:  # 原地不动或小幅度移动（惩罚）
                reward += config.no_move_penalty
                reward_details["no_move"] = 1

            self.prev_player_x = curr_player_x

        # === 敌人击杀奖励 ===
        # 1. 击杀 mob（普通敌人）
        if prev_frame is not None:
            for prev_box in prev_objects.get("mob", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("mob", []), config.iou_threshold
                ):
                    if self._check_pixel_change(
                        prev_frame, curr_frame, prev_box,
                        config.pixel_change_threshold
                    ):
                        reward += config.kill_mob
                        reward_details["kill_mob"] += 1

            # 2. 击杀 turret（炮台）
            for prev_box in prev_objects.get("turret", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("turret", []), config.iou_threshold
                ):
                    if self._check_pixel_change(
                        prev_frame, curr_frame, prev_box,
                        config.pixel_change_threshold
                    ):
                        reward += config.kill_turret
                        reward_details["kill_turret"] += 1

            # 3. 击中 boss-1_weakness（Boss弱点）
            # 弱点消失可能意味着成功击中
            for prev_box in prev_objects.get("boss-1_weakness", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("boss-1_weakness", []),
                    config.iou_threshold
                ):
                    reward += config.hit_boss_weakness
                    reward_details["hit_boss_weakness"] += 1

            # 4. 击杀 boss-1（Boss）
            for prev_box in prev_objects.get("boss-1", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("boss-1", []), config.iou_threshold
                ):
                    reward += config.kill_boss
                    reward_details["kill_boss"] += 1

            # === 躲避/受伤奖励 ===
            # 5. 躲避敌人子弹（ebullet 消失）
            bullet_dodged = 0
            for prev_box in prev_objects.get("ebullet", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("ebullet", []), config.iou_threshold
                ):
                    bullet_dodged += 1

            if bullet_dodged > 0:
                # 检查是否真的躲避成功（像素变化确认子弹消失）
                for prev_box in prev_objects.get("ebullet", []):
                    if self._check_object_disappeared(
                        prev_box, curr_objects.get("ebullet", []),
                        config.iou_threshold
                    ):
                        reward += config.dodge_bullet
                        reward_details["dodge_bullet"] += 1

            # === 危险区域奖励 ===
            # 6. 掉入坑洞（pit）- 新出现的 pit 可能表示靠近危险
            #    或者玩家位置附近的 pit 消失可能表示已经跳过
            # 这里用简单的逻辑：如果检测到 pit，给予负奖励警告
            if len(curr_objects.get("pit", [])) > 0:
                # 检查是否玩家在 pit 附近（简单实现）
                # 可以通过检测 pit 的位置变化来判断
                pass  # 暂不实现，需要更复杂的逻辑

            # 7. 进入水区域（water）- 给予警告
            if len(curr_objects.get("water", [])) > len(prev_objects.get("water", [])):
                reward += config.enter_water
                reward_details["enter_water"] += 1

            # === 道具奖励 ===
            # 8. 拾取道具（item 消失）
            for prev_box in prev_objects.get("item", []):
                if self._check_object_disappeared(
                    prev_box, curr_objects.get("item", []), config.iou_threshold
                ):
                    if self._check_pixel_change(
                        prev_frame, curr_frame, prev_box,
                        config.pixel_change_threshold
                    ):
                        reward += config.pickup_item
                        reward_details["pickup_item"] += 1

        return reward, reward_details

    def _recognize_lives(
        self, frame: np.ndarray, method: str = "template"
    ) -> int:
        """
        识别当前命数

        Args:
            frame: 原始 BGR 画面
            method: 识别方法，"template" 或 "pixel"

        Returns:
            当前命数（0-3）
        """
        # 提取命数区域
        x, y, w, h = self.lives_roi
        x = max(0, min(x, self.raw_width - w))
        y = max(0, min(y, self.raw_height - h))

        roi = frame[y : y + h, x : x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if method == "template" and self.lives_template is not None:
            # 模板匹配法
            result = cv2.matchTemplate(
                roi_gray, self.lives_template, cv2.TM_CCOEFF_NORMED
            )
            threshold = 0.8

            # 统计匹配位置数量
            locations = np.where(result >= threshold)
            matches = len(list(zip(*locations)))

            # 根据匹配数量推断命数（每个小人图标代表一条命）
            # 注意：实际可能需要根据模板大小和区域大小调整
            lives = min(matches, 3)

        else:
            # 像素统计法（简单方法）
            # 二值化后统计白色像素比例
            _, binary = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(binary > 0) / (w * h)

            # 根据像素比例映射到命数
            lives = 0
            for i, threshold in enumerate(self.lives_pixel_thresholds):
                if white_ratio >= threshold:
                    lives = i + 1

            lives = min(lives, 3)

        return lives

    def _compute_lives_reward(
        self,
        current_lives: int,
        prev_lives: int,
    ) -> float:
        """
        根据命数变化计算奖励

        Args:
            current_lives: 当前命数
            prev_lives: 上一命数

        Returns:
            命数变化奖励
        """
        config = self.reward_config
        lives_change = current_lives - prev_lives

        if lives_change < 0:
            # 命数减少（死亡）
            return config.lives_decrease
        elif lives_change > 0:
            # 命数增加（加命）
            return config.lives_increase
        else:
            # 命数不变（生存鼓励）
            return config.survival_bonus

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Returns:
            (初始观察, info字典)
        """
        # 重置状态变量
        self._reset_state()

        # 重置 retro 环境
        obs_raw, info = self.retro_env.reset(seed=seed)

        # 预处理初始帧
        obs_processed = self._preprocess_frame(obs_raw)

        # 填充帧缓冲区
        for _ in range(self.frame_stack):
            self._stack_frames(obs_processed)

        # 初始化状态
        self.prev_frame_raw = obs_raw.copy()
        self.prev_objects = self._detect_objects(obs_raw)
        self.prev_lives = self._recognize_lives(obs_raw)

        return self.frame_buffer.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个动作

        Args:
            action: 动作索引

        Returns:
            (观察, 奖励, terminated, truncated, info)
        """
        # 执行动作
        obs_raw, retro_reward, terminated, truncated, info = self.retro_env.step(
            action
        )
        self.current_step += 1

        # 超过最大步数则截断 episode
        if self.current_step >= self.max_episode_steps:
            truncated = True

        # 预处理当前帧
        obs_processed = self._preprocess_frame(obs_raw)

        # 更新帧堆叠
        stacked_obs = self._stack_frames(obs_processed)

        # 检测当前帧所有对象（每 detect_interval 帧检测一次，中间帧复用）
        if self.current_step % self.detect_interval == 0:
            curr_objects = self._detect_objects(obs_raw)
        else:
            curr_objects = self.prev_objects  # 复用上次检测结果

        # 计算对象相关奖励
        object_reward, reward_details = self._compute_rewards_from_objects(
            self.prev_frame_raw,
            obs_raw,
            self.prev_objects,
            curr_objects,
        )

        # 识别当前命数
        curr_lives = self._recognize_lives(obs_raw)

        # 计算命数变化奖励
        lives_reward = self._compute_lives_reward(curr_lives, self.prev_lives)

        # 总奖励
        total_reward = object_reward + lives_reward

        # 更新状态
        self.prev_frame_raw = obs_raw.copy()
        self.prev_objects = curr_objects
        self.prev_action = action
        self.prev_lives = curr_lives

        # 更新统计
        self.total_kills += reward_details["kill_mob"] + reward_details["kill_turret"]

        # 更新 info 字典
        info["lives"] = curr_lives
        info["total_kills"] = self.total_kills
        info["reward_details"] = reward_details
        info["object_counts"] = {
            name: len(boxes) for name, boxes in curr_objects.items()
        }
        info["retro_reward"] = retro_reward

        return stacked_obs, total_reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        """渲染环境"""
        return self.retro_env.render()

    def close(self):
        """关闭环境"""
        try:
            self.retro_env.close()
        except AttributeError:
            # 忽略 pyglet 兼容性问题
            pass


def create_contra_env(
    yolo_model_path: str,
    lives_template_path: Optional[str] = None,
    reward_config: Optional[RewardConfig] = None,
    **kwargs,
) -> ContraVisionEnv:
    """
    创建魂斗罗视觉环境的便捷函数

    Args:
        yolo_model_path: YOLO 模型路径
        lives_template_path: 命数模板路径
        reward_config: 奖励配置，若为 None 则使用默认配置
        **kwargs: 其他参数

    Returns:
        ContraVisionEnv 实例
    """
    return ContraVisionEnv(
        yolo_model_path=yolo_model_path,
        lives_template_path=lives_template_path,
        reward_config=reward_config,
        **kwargs,
    )


if __name__ == "__main__":
    # 简单测试
    import os

    # 替换为实际的 YOLO 模型路径
    yolo_path = "models/yolo-0327.pt"

    if os.path.exists(yolo_path):
        # 创建奖励配置（可自定义）
        reward_config = RewardConfig(
            kill_mob=1.0,
            kill_turret=1.5,
            hit_boss_weakness=5.0,
            kill_boss=10.0,
            dodge_bullet=0.3,
            pickup_item=2.0,
        )

        env = ContraVisionEnv(
            yolo_model_path=yolo_path,
            reward_config=reward_config,
        )
        obs, info = env.reset()
        print(f"观察空间形状: {obs.shape}")
        print(f"动作空间: {env.action_space}")
        print(f"检测到的类别:")
        for name, id_ in env.class_ids.items():
            print(f"  - {name}: ID={id_}")
        print(f"初始命数: {info.get('lives', 'N/A')}")

        # 运行几步测试
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            print(
                f"Step {i}: reward={reward:.4f}, lives={info['lives']}, "
                f"kills={info['total_kills']}, objects={info['object_counts']}"
            )
            if term or trunc:
                break

        env.close()
    else:
        print(f"YOLO 模型文件不存在: {yolo_path}")
        print("请确保已训练好 YOLO 模型并指定正确路径")