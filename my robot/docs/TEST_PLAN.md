# 具身智能仿真机器人 — 测试计划

> 配套文档：[PROJECT.md](./PROJECT.md)
> 更新日期：2026-05-16

---

## 一、测试策略总览

### 1.1 测试金字塔

```
           ┌─────────────┐
           │  E2E 场景测试 │   ← 每个 Phase 的 Demo 验收
           │   (少量,慢)    │
          ┌┴─────────────┴┐
          │   集成测试      │   ← 模块间通信、pipeline 串联
          │   (中等,中速)   │
         ┌┴───────────────┴┐
         │    单元测试       │   ← 每个函数/类的正确性
         │   (大量,快速)     │
        ┌┴─────────────────┴┐
        │   静态检查 + Lint   │   ← pre-commit: mypy, ruff, black
        └───────────────────┘
```

### 1.2 测试类型矩阵

| 测试类型 | 框架/工具 | 运行频率 | 预期耗时 | 环境 |
|---------|----------|---------|---------|------|
| 静态检查 | ruff, mypy, black | 每次 commit | < 5s | 本地 + CI |
| 单元测试 | pytest | 每次 commit | < 30s | 本地 + CI |
| 数值精度测试 | pytest + numpy | 每次 push | < 60s | CI |
| 集成测试 | pytest + ROS 2 launch_test | 每次 push | < 5min | CI (GPU) |
| 仿真一致性测试 | pytest + MuJoCo | 每日 | < 10min | AutoDL |
| E2E 场景测试 | 自定义 harness | 每 Phase 结束 | < 30min | AutoDL (5090) |
| 性能基准测试 | pytest-benchmark | 每周 | < 15min | AutoDL (5090) |
| 云环境连通性 | 自定义脚本 | 每次实例启动 | < 30s | AutoDL |

---

## 二、目录结构

```
tests/
├── conftest.py                  # 全局 fixtures: ROS 节点, 仿真实例, 测试数据
├── unit/                        # 单元测试
│   ├── perception/
│   │   ├── test_detection.py    # 目标检测: YOLO 推理, bbox 解析
│   │   ├── test_segmentation.py # 语义/实例分割: mask 精度
│   │   ├── test_fusion.py       # 多模态融合: RGB-D 对齐
│   │   └── test_camera.py       # 相机模型: 内外参, 投影/反投影
│   ├── planning/
│   │   ├── test_task_planner.py # LLM 输出解析, 技能序列生成
│   │   ├── test_motion_planner.py # RRT*, TrajOpt, 碰撞检测
│   │   └── test_path_utils.py   # 路径平滑, 插值
│   ├── control/
│   │   ├── test_ik_solver.py    # 逆运动学: 解析解/数值解
│   │   ├── test_manipulation.py # 夹爪控制, 力控
│   │   └── test_trajectory.py   # 轨迹插补: 梯形/五次多项式
│   └── policy/
│       ├── test_env.py          # Gym 环境: obs/action 空间
│       └── test_models.py       # 策略网络: 输入输出维度
├── integration/                 # 集成测试
│   ├── test_ros2_comms.py       # ROS 2 话题/服务/动作通信
│   ├── test_sensor_pipeline.py  # 传感器 → 感知 → ROS topic
│   ├── test_manip_stack.py      # 检测 → 抓取 → IK → 控制 (操作链)
│   ├── test_llm_planner.py      # 指令 → LLM → 技能调用
│   ├── test_sim_ros_bridge.py   # 仿真 ↔ ROS 2 时钟/数据同步
│   └── test_cloud_connectivity.py # SSH/Docker/GPU/端口连通性
├── simulation/                  # 仿真专项测试
│   ├── test_physics.py          # 物理正确性: 重力, 碰撞, 摩擦
│   ├── test_rendering.py        # 渲染一致性: 图像尺寸, 帧率
│   ├── test_sensors.py          # 传感器噪声模型, 数据格式
│   ├── test_robot_model.py      # URDF/MJCF: 关节限位, 质量, 惯量
│   └── test_domain_randomization.py # 域随机化参数范围
├── e2e/                         # 端到端场景测试
│   ├── phase0/
│   │   ├── test_teleop.py       # 遥操作: 关节/末端运动
│   │   └── test_sensor_stream.py # 传感器数据流完整性
│   ├── phase1/
│   │   └── test_pick_demo.py    # "拿起苹果" → 成功抓取
│   ├── phase2/
│   │   └── test_long_task.py    # 多步骤操作任务
│   ├── phase3/
│       ├── test_rl_policy.py    # RL 策略部署一致性
│       └── test_il_policy.py    # IL 策略成功率
├── performance/                 # 性能测试
│   ├── test_inference_latency.py # 感知/VLM 推理延迟
│   ├── test_control_freq.py     # 控制回路频率 (≥100Hz?)
│   ├── test_sim_realtime.py     # 仿真实时因子 (RTF ≥ 0.95?)
│   └── test_gpu_utilization.py  # RTX 5090 利用率
├── fixtures/                    # 测试固件
│   ├── robot_models/            # 简化机器人模型 (用于单元测试)
│   ├── scenes/                  # 测试场景 (空场景, 障碍物, 桌面)
│   ├── sensor_data/             # 录制的传感器数据 (rosbag)
│   ├── llm_responses/           # Mock LLM 返回 (确定性测试)
│   └── reference_trajectories/  # 参考轨迹 (Ground Truth)
└── scripts/                     # 测试辅助脚本
    ├── run_phase_tests.sh       # 按 Phase 运行全部测试
    ├── check_cloud_env.sh       # 云环境快速健康检查
    └── benchmark_runner.py      # 性能基准批量运行
```

---

## 三、单元测试规范

### 3.1 感知模块

#### test_detection.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-DET-001 | 单目标检测 | 合成 RGB 图像 (512×512)，内含 1 个苹果 | bbox 面积 IoU ≥ 0.85，类别="apple" | 数值断言 |
| UT-DET-002 | 多目标检测 | 合成图像含 3 类物体各 2 个 | 检出数 ≥ 5，分类准确率 ≥ 90% | 数值断言 |
| UT-DET-003 | 遮挡场景 | 50% 遮挡的杯子 | 仍能检出（置信度 ≥ 0.3） | 阈值断言 |
| UT-DET-004 | 空场景 | 纯色背景图像 | 检出 0 个目标 | 精确断言 |
| UT-DET-005 | 极端光照 | 过曝/欠曝图像 | 不崩溃，输出合理置信度 | 健壮性 |
| UT-DET-006 | 深度对齐 | RGB + Depth 图像对（同相机） | RGB 像素 → 3D 点误差 < 2cm | 数值精度 |

#### test_segmentation.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-SEG-001 | 语义分割 | 桌面场景 RGB 图 | mIoU ≥ 0.7 (vs GT) | 数值断言 |
| UT-SEG-002 | 实例分割 | 同类别多实例 (3个苹果) | 实例数 = 3，每个 mask IoU ≥ 0.7 | 数值断言 |
| UT-SEG-003 | 小目标 | 远距离物体 (< 32×32 px) | 不丢失，IoU ≥ 0.5 | 宽松断言 |
| UT-SEG-004 | 边界精度 | 物体边缘像素 | 边界 F1 ≥ 0.7 | 数值断言 |

#### test_fusion.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-FUS-001 | RGB-D 对齐 | 标定好的 RGB + Depth | 重投影误差 < 1 pixel | 数值精度 |
| UT-FUS-002 | 多模态时间同步 | 时间戳不同的 RGB/Depth/Lidar | 选择最近帧，同步误差 < 10ms | 时间断言 |
| UT-FUS-003 | 传感器丢帧 | 每 10 帧丢 1 帧 | 不崩溃，插值填充或跳过 | 健壮性 |

### 3.2 规划模块

#### test_task_planner.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-TP-001 | 简单指令解析 | "拿起苹果" | skill_seq = ["pick("apple")"] | 结构断言 |
| UT-TP-002 | 复合指令分解 | "拿起杯子放到盒子里" | skill_seq 长度 = 2 (pick→place) | 结构断言 |
| UT-TP-003 | 含条件指令 | "如果看到苹果就放到盘子里" | skill_seq 含条件分支 | 结构断言 |
| UT-TP-004 | 不合理指令 | "让杯子飞起来" | 返回错误码 + 说明，不崩溃 | 错误处理 |
| UT-TP-005 | LLM 输出格式错误 | 非 JSON / 缺失字段 | 兜底解析，返回安全默认值 | 容错性 |
| UT-TP-006 | 长指令截断 | 超过 token 限制的指令 | 优雅降级，truncate + 提示 | 边界条件 |

#### test_motion_planner.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-MP-001 | 无障碍直线 | start→goal 无遮挡 (6-DoF) | 找到路径，路径点数 ≥ 2，无碰撞 | 断言 |
| UT-MP-002 | 含障碍物 | 场景有 3 个障碍物 | 路径全程无碰撞 (最近距离 > 0) | 碰撞检测 |
| UT-MP-003 | 不可达目标 | goal 在障碍物内部 | 返回失败 + 最近可行点，不无限循环 | 超时保护 |
| UT-MP-004 | 规划超时 | 极复杂迷宫场景 | 在 timeout 内返回 best-effort 路径 | 超时断言 |
| UT-MP-005 | 关节限位 | goal 超出关节范围 | 返回失败或 clamp 到限位内 | 边界断言 |
| UT-MP-006 | 路径平滑 | 原始路径有锯齿 | 平滑后曲率连续，与原始偏差 < 2cm | 数值断言 |

### 3.3 控制模块

#### test_ik_solver.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-IK-001 | 可达位姿 | 末端位姿在工作空间内 | IK 解存在，正解回代误差 < 1e-3 | 数值精度 |
| UT-IK-002 | 奇异点附近 | 末端位姿接近奇异构型 | 不崩溃，返回最近可行解或警告 | 数值稳定性 |
| UT-IK-003 | 多解选择 | 存在多个 IK 解 | 选择关节角变化最小的解 | 最优性 |
| UT-IK-004 | 不可达位姿 | 末端在工作空间外 | 返回错误 + 最近可达位姿 | 错误处理 |
| UT-IK-005 | 碰撞约束 | 某解会碰撞 | 在有碰撞的解存在时仍返回无碰撞解 | 约束满足 |

#### test_trajectory.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-TRJ-001 | 梯形速度插补 | via points × 5 | 速度/加速度在限幅内 | 约束满足 |
| UT-TRJ-002 | 五次多项式 | start + end 位姿 | 位置/速度/加速度连续 | 平滑性 |
| UT-TRJ-003 | 时间最优 | 路径 + 动力学约束 | 完成时间 ≤ 理论下限 × 1.2 | 最优性 |
| UT-TRJ-004 | 在线重规划 | 中途变更目标 | 新轨迹与原轨迹 C¹ 连接 | 连续性 |

### 3.4 策略模块

#### test_env.py

| 用例ID | 测试场景 | 输入 | 期望输出 | 验证方式 |
|--------|---------|------|---------|---------|
| UT-ENV-001 | obs space 类型/维度 | 调用 env.observation_space | shape 匹配配置，dtype=float32 | 接口断言 |
| UT-ENV-002 | action space 范围 | 调用 env.action_space | low/high 在合理范围内 | 接口断言 |
| UT-ENV-003 | reset 一致性 | 连续 10 次 reset() | obs 维度一致，agent 在初始分布内 | 一致性 |
| UT-ENV-004 | step 返回值 | step(action) 调用 | (obs, reward, terminated, truncated, info) | 协议断言 |
| UT-ENV-005 | reward 范围 | 多次 step | reward 有界，无 NaN/Inf | 数值合法性 |

---

## 四、集成测试

### 4.1 ROS 2 通信

| 用例ID | 测试场景 | 验证点 | 判定标准 |
|--------|---------|-------|---------|
| IT-ROS-001 | 话题发布/订阅 | 感知节点 pub → 规划节点 sub | 延迟 < 5ms，无丢帧 |
| IT-ROS-002 | 服务调用 | 规划节点请求 IK 服务 | 响应时间 < 10ms |
| IT-ROS-003 | Action 执行 | pick Action (含 feedback) | Feedback 频率 ≥ 10Hz |
| IT-ROS-004 | 多节点并发 | 5 个节点同时通信 | 无死锁、无数据竞争 |
| IT-ROS-005 | 节点崩溃恢复 | 杀掉一个节点 | 其他节点检测到并降级运行 |

### 4.2 Pipeline 端到端 (模块串联)

| 用例ID | 测试场景 | 数据流 | 判定标准 |
|--------|---------|-------|---------|
| IT-PIPE-001 | 传感器→ROS | Camera → RGB-D topic | 帧率 ≥ 30Hz, 分辨率正确 |
| IT-PIPE-002 | 感知→规划 | Detection bbox → Grasp pose | 目标 3D 坐标误差 < 5cm |
| IT-PIPE-003 | 规划→控制 | Path waypoints → Joint cmd | 控制频率 ≥ 100Hz |
| IT-PIPE-004 | 仿真↔ROS 同步 | Sim 时钟 ↔ ROS 时钟 | 时钟偏差 < 1ms |
| IT-PIPE-005 | VLM→技能调用 | 自然语言 → 技能执行 | 技能名称+参数正确解析 |

### 4.3 云环境连通性

| 用例ID | 测试场景 | 验证点 | 判定标准 |
|--------|---------|-------|---------|
| IT-CLD-001 | SSH 连接 | AutoDL 实例可达 | 建立连接 < 5s |
| IT-CLD-002 | GPU 可用 | `nvidia-smi` | RTX 5090 显存 ≥ 30GB 可用 |
| IT-CLD-003 | Docker 运行 | Isaac Sim 容器启动 | 容器 running，端口 8211 可访问 |
| IT-CLD-004 | Streaming Client | 浏览器访问 8211 | 返回 Isaac Sim 页面 |
| IT-CLD-005 | 数据盘挂载 | `/root/autodl-tmp` 读写 | 读写速度 ≥ 200 MB/s |

---

## 五、仿真专项测试

### 5.1 物理正确性

| 用例ID | 测试场景 | 验证点 | 容差 |
|--------|---------|-------|------|
| SIM-PHY-001 | 自由落体 | 物体从 1m 落下，落地时间 | 与理论值偏差 < 1% |
| SIM-PHY-002 | 弹性碰撞 | 两球对心碰撞 | 动量守恒偏差 < 0.1% |
| SIM-PHY-003 | 摩擦斜面 | 物体在 30° 斜面滑动 | 加速度偏差 < 5% |
| SIM-PHY-004 | 关节力矩 | 施加力矩，测量角加速度 | 与模型偏差 < 1% |
| SIM-PHY-005 | 接触力 | 夹爪抓取，测接触力 | 力传感器读数稳定（噪声 < 0.1N） |

### 5.2 渲染一致性

| 用例ID | 测试场景 | 验证点 | 判定标准 |
|--------|---------|-------|---------|
| SIM-RND-001 | 图像尺寸 | Camera sensor 输出 | 分辨率 = 配置值 (640×480) |
| SIM-RND-002 | 帧率 | 连续 100 帧计时 | FPS ≥ 30 (headless RTX 5090) |
| SIM-RND-003 | 深度精度 | Depth GT vs 渲染深度 | RMSE < 1mm (近距) |
| SIM-RND-004 | 语义标签 | Segmentation sensor | 像素级标签与 GT 一致 |

### 5.3 机器人模型

| 用例ID | 测试场景 | 验证点 | 判定标准 |
|--------|---------|-------|---------|
| SIM-ROB-001 | 关节限位 | 尝试驱动 UR5 关节超限 (±360° 范围) | 被限位器阻止，不穿透 |
| SIM-ROB-002 | 自碰撞 | 臂与桌面/自身碰撞 | 检测到碰撞并停止/规避 |
| SIM-ROB-003 | 质量/惯量 | 计算 UR5 总质量 vs URDF 标称 | 偏差 < 0.1% |
| SIM-ROB-004 | 末端精度 | IK 解对应的关节角驱动后 | 末端位姿误差 < 1mm + 0.1° |
| SIM-ROB-005 | IK 闭式解 | 随机采样 100 个工作空间内位姿 | 100% 有解，正解回代误差 < 1e-3 |

---

## 六、E2E 场景测试（按 Phase 门禁）

### Phase 0 门禁 — 环境就绪

| 用例ID | 测试场景 | 操作步骤 | 验收标准 |
|--------|---------|---------|---------|
| E2E-P0-001 | 遥操作闭环 | 键盘发送关节/末端指令 → 机械臂在仿真中运动 | 延迟 < 200ms，运动方向正确 |
| E2E-P0-002 | 传感器数据流 | 启动仿真，订阅所有 sensor topic | 每种传感器都有数据，无超时 |
| E2E-P0-003 | MuJoCo 基本场景 | 启动 MuJoCo 场景，step 1000 次 | 不崩溃，物理无明显异常 |

**P0 出口标准：全部 PASS 方可进入 Phase 1**

### Phase 1 门禁 — 物体抓取

| 用例ID | 测试场景 | 操作步骤 | 验收标准 |
|--------|---------|---------|---------|
| E2E-P1-001 | 孤立物体抓取 | 桌上单个苹果，输入"拿起苹果" | 成功抓取并提升 ≥ 10cm |
| E2E-P1-002 | 多物体选择 | 桌上有苹果+杯子，输入"拿起杯子" | 抓取正确目标 |
| E2E-P1-003 | 不同形状 | 依次抓取球/方块/圆柱 | 每种形状成功率 ≥ 70% |
| E2E-P1-004 | 放置 | 抓取后输入"放到收纳盒里" | 物体稳定放置在目标位置 |
| E2E-P1-005 | 抓取失败恢复 | 首次抓取滑脱 | 检测失败，自动重试 ≥ 1 次 |

**P1 出口标准：抓取成功率 ≥ 75% (≥ 15/20)**

### Phase 2 门禁 — 长任务执行

| 用例ID | 测试场景 | 操作步骤 | 验收标准 |
|--------|---------|---------|---------|
| E2E-P2-001 | 双步任务 | "拿苹果放到盒子里" | pick→place 全成功 |
| E2E-P2-002 | 三步任务 | "把杯子放进收纳盒，再把苹果放到盘子里" | 完成所有子任务 |
| E2E-P2-003 | 任务中断恢复 | 人为移动目标物体 | 检测环境变化，重规划并继续 |
| E2E-P2-004 | 含条件任务 | "如果看到苹果就放到红盘子里，否则把杯子放到蓝盘子里" | 根据场景正确分支 |
| E2E-P2-005 | 修正指令 | 执行中追加"不对，放到右边那个盒子" | 根据新指令重规划 |

**P2 出口标准：3 步内任务成功率 ≥ 60% (≥ 6/10)**

### Phase 3 门禁 — 学习策略

| 用例ID | 测试场景 | 操作步骤 | 验收标准 |
|--------|---------|---------|---------|
| E2E-P3-001 | RL 策略部署 | 训练好的 checkpoint → 仿真中运行 | 成功率与训练末段一致 (±5%) |
| E2E-P3-002 | 域随机化测试 | 改变光照/纹理后运行同一策略 | 成功率下降 ≤ 15% |
| E2E-P3-003 | 策略切换 | 在 RL 策略和规则策略间切换 | 切换平滑，无抖动 |
| E2E-P3-004 | IL 策略泛化 | 未见过的新物体放置 | 成功率 ≥ 50% (首次) |

**P3 出口标准：RL 策略成功率 ≥ 80%，泛化下降 ≤ 20%**

### Phase 4 门禁 — 高级能力

| 用例ID | 测试场景 | 验收标准 |
|--------|---------|---------|
| E2E-P4-001 | 灵巧操作 | 单手旋转物体 ≥ 90° |
| E2E-P4-002 | 接触密集型 | 推/拉/开门操作，力控稳定 |
| E2E-P4-003 | 双臂协作 | 双臂协同搬运，不碰撞 |
| E2E-P4-004 | 开放词汇 | 未见过的物体名称 + 新指令 |
| E2E-P4-005 | Sim2Real | 真实机械臂执行仿真训练策略，成功率 ≥ 50% |

---

## 七、性能基准测试

### 7.1 关键性能指标 (KPI)

| 指标 | 目标值 | 测量方法 | 不达标处理 |
|------|-------|---------|-----------|
| 实时因子 (RTF) | ≥ 0.95 | 仿真时间 / 墙钟时间 | 简化场景或降低渲染质量 |
| 控制频率 | ≥ 100 Hz | 控制回调间隔统计 | 优化 IK 求解器或降频 |
| 感知推理延迟 | < 50 ms | 图像入 → 检测结果出 | 减少分辨率 / 量化模型 |
| VLM 推理延迟 | < 2 s | prompt → response 端到端 | 换用小模型 / caching |
| GPU 显存占用 | < 28 GB | `nvidia-smi` 峰值 | 减小 batch / fp16 |
| ROS 2 通信延迟 | < 5 ms | node pub ↔ node sub | 调整 QoS / 减少 topic |

### 7.2 基准回归测试

| 用例ID | 测量对象 | 回归阈值 | 触发条件 |
|--------|---------|---------|---------|
| PERF-001 | 感知推理时间 | 较基准增加 > 20% | 模型或依赖变更 |
| PERF-002 | 控制回路频率 | 低于 100Hz | 控制代码变更 |
| PERF-003 | 仿真 RTF | 低于 0.9 | 场景或物理参数变更 |
| PERF-004 | 内存泄漏 | 运行 30min 内存增长 > 500MB | 每次 Phase 出口测试 |

---

## 八、测试数据与 Mock 策略

### 8.1 Mock 原则

- **单元测试**：Mock 所有外部依赖（LLM API, 仿真环境, ROS 节点）
- **集成测试**：使用真实 ROS 2 + Mock 的 LLM 响应
- **E2E 测试**：全链路真实，仅 Mock 不可控的外部 API (GPT/Claude)

### 8.2 测试数据集

| 数据集 | 用途 | 来源 |
|--------|------|------|
| 合成 RGB 图像 | 检测/分割单元测试 | `tests/fixtures/sensor_data/` 脚本生成 |
| 录制 rosbag | 集成测试回放 | 仿真中录制标准场景 |
| Mock LLM 响应 | 任务规划确定性测试 | `tests/fixtures/llm_responses/*.json` |
| 参考轨迹 | 控制精度基准 | 离线计算的理论最优轨迹 |
| 标杆场景 | 性能基准 | 固定仿真场景 (物体数量/位置固定) |

### 8.3 LLM Mock 示例

```python
# tests/fixtures/llm_responses/pick_apple.json
{
  "input": "拿起苹果",
  "output": {
    "task_type": "manipulation",
    "skill_sequence": [
      {"skill": "pick", "params": {"object": "apple", "frame": "camera"}}
    ],
    "confidence": 0.95
  }
}
```

---

## 九、CI/CD 集成 (未来)

> 当前阶段（AutoDL 云环境）以手动 + 脚本执行为主，后续可迁移至 GitHub Actions + 自托管 Runner。

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│ Commit  │───▶│  Lint +  │───▶│  Unit Tests  │───▶│ 结果通知   │
│ / Push  │    │  Type    │    │  (pytest)    │    │  (控制台)  │
└─────────┘    └──────────┘    └──────────────┘    └────────────┘
                                      │
                    ┌─────────────────┘
                    ▼
              ┌──────────┐    ┌──────────────┐
              │ Integ.   │───▶│ E2E (Nightly)│
              │ Tests    │    │ on AutoDL    │
              └──────────┘    └──────────────┘
```

**Pre-commit 配置（`.pre-commit-config.yaml`）：**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

---

## 十、测试执行 SOP

### 10.1 每次开发前 — 快速检查

```bash
# 1. 云环境健康检查
bash tests/scripts/check_cloud_env.sh
# 预期: SSH ✓, GPU ✓, Docker ✓, Port 8211 ✓

# 2. 静态检查
pre-commit run --all-files

# 3. 相关模块单元测试
pytest tests/unit/<改动的模块>/ -v --tb=short
```

### 10.2 每日构建 — 完整性检查

```bash
# 全量单元 + 集成测试 (无 LLM 调用)
pytest tests/unit/ tests/integration/ \
  --mock-llm \
  -n auto \
  --timeout=30 \
  --cov=simulation/ --cov=perception/ --cov=planning/ --cov=control/ \
  --cov-report=html \
  -v
```

### 10.3 Phase 出口 — 门禁检查

```bash
# 运行当前 Phase 对应 E2E 测试
bash tests/scripts/run_phase_tests.sh <phase_number>
# 例: bash tests/scripts/run_phase_tests.sh 1

# 同时运行性能基准
pytest tests/performance/ --benchmark-only --benchmark-autosave
```

### 10.4 发版前 — 全量回归

```bash
# 完整测试套件 (需 AutoDL RTX 5090)
pytest tests/ -n auto \
  --run-e2e \
  --run-simulation \
  --timeout=600 \
  --html=report.html \
  --self-contained-html \
  -v
```

---

## 十一、Bug 分类与处理流程

| 严重级别 | 定义 | 示例 | 处理要求 |
|---------|------|------|---------|
| P0-Critical | 仿真崩溃 / 数据丢失 | 容器 OOM Kill, 磁盘满导致 rosbag 丢失 | 立即修复，阻塞任何新功能 |
| P1-High | 核心功能不可用 | IK 求解器持续返回错误解, 抓取持续失败 | 24h 内修复 |
| P2-Medium | 功能降级 | 检测偶尔漏检, 抓取成功率下降 5% | 当前 Phase 内修复 |
| P3-Low | 体验问题 | 路径不够平滑, 日志格式混乱 | 下个 Phase 修复或标记 known-issue |

**测试发现的 Bug 必须：**
1. 在测试报告中记录复现步骤
2. 关联失败测试用例 ID
3. 修复后对应的回归测试必须加入 CI

---

> **更新日志**
> - 2026-05-16 (v2): 聚焦机械臂操作，移除导航/移动底盘相关测试，Phase 重新编号
> - 2026-05-16 (v1): 初始测试计划，覆盖 Phase 0-5 全阶段