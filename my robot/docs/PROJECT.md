# 具身智能仿真机器人项目

## 一、项目概述

本项目旨在构建一个**具身智能（Embodied Intelligence）仿真机械臂系统**，在物理仿真环境中实现视觉感知、任务理解与灵巧操作的闭环。项目以仿真为起点，聚焦机械臂抓取与操作能力，降低硬件门槛的同时，保证算法可迁移至真实机器人平台。

### 核心目标

- 在仿真环境中构建机械臂操作 pipeline
- 支持视觉感知、自然语言指令理解、任务规划与灵巧操作
- 算法具备 sim-to-real 迁移能力
- 模块化设计，各组件可独立迭代

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户交互层                            │
│         自然语言指令 / 目标图像 / 遥操作输入                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   高层决策层 (Brain)                       │
│   ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│   │  VLM/LLM │  │ 任务规划  │  │  记忆与上下文管理   │   │
│   │  语义理解 │  │  Task Plan│  │  Memory & Context  │   │
│   └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   中层技能层 (Skills)                      │
│   ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│   │ 抓取姿态  │  │ 灵巧操作  │  │  环境感知与建模     │   │
│   │ Grasping │  │Dexterous │  │  Scene Understanding│   │
│   └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   底层控制层 (Control)                     │
│   ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│   │ 运动规划  │  │ 力控/柔顺 │  │  逆运动学(IK)      │   │
│   │Motion Pl │  │ Impedance│  │  轨迹插补          │   │
│   └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    仿真环境层                              │
│        Isaac Sim / MuJoCo / PyBullet / SAPIEN           │
│              传感器仿真 / 物理引擎 / 渲染                   │
└─────────────────────────────────────────────────────────┘
```

---

## 三、技术选型

| 层级 | 技术方案 | 说明 |
|------|---------|------|
| 仿真引擎 | **Isaac Sim** (主) + **MuJoCo** (辅) | Isaac Sim 提供高保真渲染与 ROS 桥接；MuJoCo 轻量快速，适合 RL 训练 |
| 机器人模型 | **UR5** (主) + Robotiq 2F-85 夹爪 | 6 DoF 机械臂，IK 有闭式解，MuJoCo/Isaac Sim 原生支持 |
| 感知 | RGB-D + 语义分割 + 目标检测 | 仿真中获取 Ground Truth，逐步引入 Sim2Real 域适应 |
| 高层推理 | GPT-4V / Claude / 本地 VLM | 自然语言→任务分解→技能调用 |
| 操作 | 6-DoF 抓取 (GraspNet) + 灵巧操作 | 从抓取到 in-hand manipulation |
| 运动控制 | 阻抗控制 / RL 策略 | 柔顺操作与接触密集型任务 |
| 云平台 | **AutoDL** (RTX 5090) | 按需租用 GPU 实例，降低硬件成本 |
| 中间件 | ROS 2 Humble | 模块间通信、消息标准化 |
| 深度学习框架 | PyTorch + Isaac Lab | 策略训练与部署 |
| 语言 | Python 3.10+ | 主开发语言 |

---

## 四、技术路线（分阶段）

### Phase 0：环境搭建与基础仿真（1-2 周）

- [ ] 租用 AutoDL 云实例（RTX 5090, Ubuntu 22.04, CUDA 12.4+）
- [ ] 配置 Isaac Sim 容器（headless 模式 + Omniverse Streaming Client 远程渲染）
- [ ] 安装 MuJoCo（pip 直接安装，无需 GUI）
- [ ] 导入 UR5 机械臂 + Robotiq 2F-85 夹爪模型
- [ ] 实现键盘/手柄遥操作（关节空间 + 末端空间）
- [ ] 验证传感器数据流（RGB, Depth, 关节状态, 力/力矩）
- [ ] 建立项目目录结构与代码规范

#### AutoDL 云环境配置

**硬件配置：**
| 项目 | 规格 |
|------|------|
| GPU | NVIDIA RTX 5090 (32GB) |
| CPU | 16 vCPU |
| 内存 | 64 GB |
| 系统盘 | 100 GB |
| 数据盘 | 200 GB（挂载到 `/root/autodl-tmp`） |
| 镜像 | Ubuntu 22.04 + CUDA 12.4 + Miniconda |

**远程访问方案：**
```
本地 VS Code ──(SSH)──▶ AutoDL 实例 ──▶ Isaac Sim (headless)
                                    ──▶ MuJoCo (headless 渲染)
                                    ──▶ ROS 2 节点
```

- **代码开发**：VS Code Remote-SSH 直连 AutoDL 实例
- **Isaac Sim GUI**：启用 Omniverse Streaming Client，浏览器访问 `http://<实例IP>:8211`
- **渲染**：Isaac Sim headless 模式利用 RTX 5090 GPU 渲染，视频流推送至本地浏览器
- **文件同步**：AutoDL 提供 JupyterLab 文件管理 + SCP/SFTP

**Isaac Sim 容器启动命令：**
```bash
# 拉取 Isaac Sim 镜像（首次）
docker pull nvcr.io/nvidia/isaac-sim:4.2.0

# 启动容器（headless + streaming）
docker run --gpus all -d \
  --name isaac-sim \
  -p 8211:8211 \
  -p 8891:8891 \
  -v /root/autodl-tmp/isaac-sim-kit:/isaac-sim/kit \
  nvcr.io/nvidia/isaac-sim:4.2.0 \
  /isaac-sim/runheadless.native.sh \
  --/app/livestream/enabled=True \
  --/app/livestream/websocket_port=8211
```

#### 项目目录结构规划

```
my-robot/
├── docs/                    # 项目文档
│   ├── PROJECT.md           # 本文件：项目架构与技术路线
│   ├── TEST_PLAN.md         # 测试计划
│   ├── PROGRESS.md          # 进度追踪
│   └── DEPENDENCIES.md      # 依赖技术文档
├── simulation/              # 仿真环境
│   ├── isaac_sim/           # Isaac Sim 场景与脚本
│   ├── mujoco/              # MuJoCo 模型与脚本
│   └── assets/              # 机器人 URDF/MJCF 模型
├── perception/              # 感知模块
│   ├── detection/           # 目标检测
│   ├── segmentation/        # 语义/实例分割
│   └── fusion/              # 多模态融合
├── planning/                # 规划模块
│   ├── task_planner/        # 任务级规划 (LLM-based)
│   └── motion_planner/      # 运动规划 (RRT, TrajOpt, 碰撞检测)
├── control/                 # 控制模块
│   ├── ik_solver/           # 逆运动学
│   └── manipulation/        # 操作控制 (力控, 夹爪)
├── policy/                  # 学习策略
│   ├── rl/                  # 强化学习训练
│   ├── il/                  # 模仿学习
│   └── checkpoints/         # 模型权重
├── ros2_ws/                 # ROS 2 工作空间
│   └── src/
├── scripts/                 # 工具脚本
├── tests/                   # 测试（详见 docs/TEST_PLAN.md）
├── requirements.txt
└── README.md
```

### Phase 1：物体操作（4-6 周）

- [ ] 集成 GraspNet / AnyGrasp 6-DoF 抓取检测
- [ ] 实现 MoveIt 2 运动规划（或自研 RRT* + TrajOpt）
- [ ] 逆运动学求解与碰撞检测
- [ ] 力控抓取仿真（阻抗控制）
- [ ] 完成首个 Demo：**"拿起桌上的苹果"**

**操作 pipeline：**

```
RGB-D → 目标检测 → 抓取姿态估计 → 运动规划 → IK → 关节控制 → 力反馈闭环
```

### Phase 2：端到端任务执行（4-6 周）

- [ ] 接入 VLM（视觉语言模型）进行场景理解
- [ ] 实现任务图（Task Graph）规划——长序列任务分解
- [ ] 构建技能原语库（Skill Primitives）：pick, place, push, pull, open, close, stack, pour
- [ ] 引入 ReAct / Code-as-Policies 范式进行推理
- [ ] 完成第二个 Demo：**"把桌上的杯子放进收纳盒，再把苹果放到盘子里"**

**任务执行流程：**

```
输入: "把桌上的杯子放进收纳盒，再把苹果放到盘子里"
    │
    ▼
VLM 场景理解 → 任务分解:
    1. pick("cup") → place("cup", "box") → pick("apple") → place("apple", "plate")
    │
    ▼
Skill Sequencer → 逐技能执行 → 每步感知验证 → 异常恢复
```

### Phase 3：学习驱动策略（6-8 周）

- [ ] 搭建 Isaac Lab 训练框架
- [ ] 实现 PPO/SAC 强化学习 pipeline
- [ ] 训练推/拉/开门等接触密集型任务
- [ ] 实现 DAgger/ACT 模仿学习
- [ ] 域随机化（Domain Randomization）提升 sim-to-real 泛化性
- [ ] 完成第三个 Demo：**从仿真中学到的策略控制真实机器人**

### Phase 4：高级能力（持续迭代）

- [ ] 灵巧手操作（Dexterous Manipulation）
- [ ] 移动操作统一控制（Whole-Body MPC）
- [ ] 多机器人协作
- [ ] 开放词汇操作（Open-Vocabulary Manipulation）
- [ ] 人机交互与安全
- [ ] 真实机器人迁移与标定

---

## 五、关键技术原理

### 5.1 仿真环境选型对比

| 特性 | Isaac Sim | MuJoCo | PyBullet | SAPIEN |
|------|-----------|--------|----------|--------|
| 渲染质量 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 物理精度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 速度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| ROS 集成 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| RL 训练 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 学习曲线 | 陡峭 | 平缓 | 平缓 | 中等 |

**建议：** 主仿真用 Isaac Sim（高保真+ROS桥接），RL训练用 MuJoCo（轻量快速），两者配合。

#### 云环境适配要点

| 场景 | Isaac Sim | MuJoCo |
|------|-----------|--------|
| GUI | Streaming Client 远程渲染（浏览器） | 不需要 GUI，纯 headless |
| GPU 利用率 | RTX 5090 渲染 + 推理并行，利用率高 | 物理计算为主，GPU 占用低 |
| Docker | 官方 NGC 容器，开箱即用 | 无需容器，pip 安装 |
| 远程开发 | VS Code Remote-SSH + 端口转发 | VS Code Remote-SSH 即可 |
| 数据持久化 | 容器内数据需挂载到 `/root/autodl-tmp` | 直接存储在数据盘 |

### 5.2 UR5 机械臂

**选型理由：**

| 特性 | UR5 | 为什么重要 |
|------|-----|-----------|
| 6 DoF（非冗余） | IK 存在闭式解（8 组） | 不需要数值优化，调试简单 |
| 工作半径 850mm | 覆盖典型桌面操作场景 | 满足抓取/放置需求 |
| 负载 5kg | 典型物体操作足够 | 杯子/水果/工具均在范围内 |
| MuJoCo 官方模型 | `mujoco_menagerie` 直接加载 | 零配置启动 |
| Isaac Sim 原生支持 | 内置 UR5 USD 资产 | GUI 拖拽即用 |
| Robotiq 2F-85 夹爪 | 平行夹爪，控制简单 | 仅需 1 DoF 控制 vs 灵巧手的 16+ DoF |
| 社区资料丰富 | GitHub / 论文 / 教程极多 | 遇到问题有现成答案 |

**UR5 DH 参数：**

| 关节 | a (m) | d (m) | α (rad) | 关节范围 |
|------|-------|-------|---------|---------|
| 1 (base) | 0 | 0.0892 | π/2 | ±360° |
| 2 (shoulder) | -0.425 | 0 | 0 | ±360° |
| 3 (elbow) | -0.392 | 0 | 0 | ±360° |
| 4 (wrist 1) | 0 | 0.1093 | π/2 | ±360° |
| 5 (wrist 2) | 0 | 0.09475 | -π/2 | ±360° |
| 6 (wrist 3) | 0 | 0.0825 | 0 | ±360° |

> IK 闭式解推导参见 [UR5 Kinematics](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)。6 个关节均为旋转关节，前 3 个决定末端位置，后 3 个决定末端姿态，存在解耦结构使得闭式解可能。

**MuJoCo 加载示例：**

```python
import mujoco
import mujoco.viewer

# 从 mujoco_menagerie 加载 UR5 + Robotiq 夹爪
model = mujoco.MjModel.from_xml_path("ur5e/ur5e.xml")
data = mujoco.MjData(model)

# 直接驱动关节角
data.ctrl[:6] = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]  # 6 个关节
mujoco.mj_step(model, data)
```

### 5.4 VLM 驱动的任务规划

采用层次化推理框架：

1. **场景理解**：VLM 分析当前观测 → 输出场景图（物体、位置、关系）
2. **任务分解**：LLM 将自然语言指令 → 分解为技能序列
3. **技能执行**：每个技能由专门的策略模型执行
4. **闭环验证**：每步执行后进行视觉验证，失败则重规划

参考范式：
- **SayCan** (Google): LLM + Affordance
- **Code-as-Policies** (Google): LLM → Python 代码生成
- **RT-2** (Google): VLM 端到端输出动作
- **Octo** (Berkeley): 通用机器人策略模型

### 5.5 Sim-to-Real 迁移策略

| 方法 | 说明 |
|------|------|
| 域随机化 | 训练时随机化光照、纹理、物理参数 |
| 域适配 | 使用 GAN/DAE 对齐仿真与真实图像分布 |
| 系统辨识 | 标定真实机器人的动力学参数，反向校准仿真 |
| 渐进迁移 | 先在仿真训练 → 少量真实数据微调 |

---

## 六、依赖库清单

> 完整依赖技术文档（含版本号、安装脚本、兼容性矩阵）：**[DEPENDENCIES.md](./DEPENDENCIES.md)**

### 当前 Phase 0 运行时依赖

```txt
mujoco>=3.2.2,<4.0       # MuJoCo 物理引擎
numpy>=2.0,<3.0           # 数值计算
opencv-python>=4.10.0,<5.0 # 图像处理
pynput>=1.7.7,<2.0        # 键盘输入（遥操作）
pytest>=8.3,<9.0          # 单元测试
```

### 后续 Phase 关键新增依赖

| Phase | 新增依赖 | 用途 |
|-------|---------|------|
| Phase 1 | torch 2.6.0, torchvision 0.21.0, open3d 0.19.0, scipy 1.15.0, pin 3.2.0, casadi 3.6.7 | 感知、规划与控制 |
| Phase 2 | transformers 4.51.0, accelerate 1.4.0, httpx 0.28.1 | VLM/LLM 任务规划 |
| Phase 3 | dm_control 1.0.25, stable-baselines3 2.5.0, Isaac Lab 2.1.0 | RL/IL 策略训练 |
| Phase 4 | trimesh 4.5.0, cvxopt 1.3.2, networkx 3.4 | 灵巧操作与多机协作 |

### 系统环境

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 22.04 LTS | AutoDL 镜像 |
| Python | **3.12.9** | 主开发语言 |
| CUDA | 12.4.1 | RTX 5090 GPU 驱动 |
| ROS 2 | Humble LTS | 中间件通信 |
| Isaac Sim | 4.5.0 (NGC Docker) | 仿真平台 |

---

## 七、里程碑

| 里程碑 | 时间 | 产出 |
|--------|------|------|
| M0: 环境跑通 | Week 1-2 | 机械臂在仿真中能动 |
| M1: 抓取操作 | Week 3-8 | 视觉引导的物体抓取 |
| M2: 长任务执行 | Week 9-14 | 多步骤操作任务自主完成 |
| M3: 学习策略 | Week 15-22 | RL 训练的策略驱动操作 |
| M4: 高级能力 | Week 23+ | 灵巧操作、Sim2Real 迁移 |

---

## 八、参考资料

- [NVIDIA Isaac Sim 文档](https://docs.omniverse.nvidia.com/isaacsim/)
- [Isaac Lab (原 Orbit)](https://isaac-sim.github.io/IsaacLab/)
- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)
- [ROS 2 文档](https://docs.ros.org/en/humble/)
- [SayCan: Do As I Can, Not As I Say](https://say-can.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
- [Octo: An Open-Source Generalist Robot Policy](https://octo-models.github.io/)
- [GraspNet](https://graspnet.net/)
- [MoveIt 2](https://moveit.ros.org/)
- [测试计划文档](./TEST_PLAN.md)

---

> **更新日志**
> - 2026-05-16 (v3): 聚焦机械臂操作，移除导航/移动底盘，Phase 重新编号为 0→1→2→3→4
> - 2026-05-16 (v2): 新增 AutoDL 云环境 (RTX 5090) 部署方案、远程开发架构
> - 2026-05-16 (v1): 初始版本，项目架构设计与技术路线规划