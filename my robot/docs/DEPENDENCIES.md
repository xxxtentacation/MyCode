# 项目依赖技术文档

> 配套：[PROJECT.md](./PROJECT.md) | [PROGRESS.md](./PROGRESS.md) | [TEST_PLAN.md](./TEST_PLAN.md)
> 更新：2026-05-29
> Python 版本：**3.12**

---

## 一、系统环境

| 组件 | 版本/规格 | 说明 |
|------|----------|------|
| 操作系统 | Ubuntu 22.04 LTS | AutoDL 云实例镜像 |
| Python | **3.12.9** | 主开发语言，Python 3.12 最新稳定版 |
| CUDA | **12.4.1** | RTX 5090 驱动支持，Isaac Sim 4.2.0 要求 |
| GPU | NVIDIA RTX 5090 (32 GB) | AutoDL 云实例 |
| Docker | **27.3.1** | Isaac Sim 容器运行环境 |
| GCC | **11.4.0** | 编译 PyTorch/CasADi 等 C++ 扩展 |

---

## 二、Python 版本策略

| 项目 | 版本 | 说明 |
|------|------|------|
| Python | **3.12.9** | 主版本，所有依赖以此为准 |
| pip | **24.3.1** | Python 3.12 自带 |
| virtualenv / venv | 标准库 | 虚拟环境管理建议使用 `python3.12 -m venv` |
| Miniconda | **24.11.0** | AutoDL 预装，conda 环境隔离（备选） |

> Python 3.12 关键特性：类型提示语法改进 (`type` 语句, PEP 695)、GIL 可选 (PEP 684, 需从源码编译)、性能提升约 5%。

---

## 三、Phase 0 — 当前依赖（已实现）

### 3.1 核心运行时依赖

| 包名 | 版本 | 用途 | 使用模块 |
|------|------|------|---------|
| **mujoco** | **3.2.2** | MuJoCo 物理引擎，机器人动力学仿真 | `simulation/mujoco_env.py` |
| **numpy** | **2.1.3** | 数值计算、矩阵运算、线性代数 | 全部模块 |
| **opencv-python** | **4.10.0** | 图像处理、相机内参矩阵计算 | `perception/camera.py` |
| **pynput** | **1.7.7** | 键盘输入监听，遥操作控制 | `scripts/teleop.py` |

### 3.2 当前 requirements.txt

```txt
mujoco>=3.2.2,<4.0
numpy>=2.0,<3.0
opencv-python>=4.10.0,<5.0
pynput>=1.7.7,<2.0
```

### 3.3 Phase 0 测试依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| **pytest** | **8.3.4** | 单元测试框架，`tests/unit/control/test_ik_solver.py` |

---

## 四、Phase 1 — 物体操作（计划依赖）

### 4.1 感知

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **torch** | **2.6.0** | 深度学习框架，GraspNet 推理 | 匹配 CUDA 12.4，支持 RTX 5090 (sm_120) |
| **torchvision** | **0.21.0** | 图像预处理、数据增强 | 与 torch 2.6.0 配套 |
| **open3d** | **0.19.0** | 点云处理、ICP 配准、TSDF 融合 | RGB-D → 点云 → 场景重建 |

### 4.2 规划与控制

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **scipy** | **1.15.0** | 科学计算：优化、插值、空间变换 | 运动规划数值求解 |
| **pin** | **3.2.0** | Pinocchio 刚体动力学库 | 正/逆动力学、雅可比矩阵（替代手工 DH 计算） |
| **casadi** | **3.6.7** | 非线性优化框架 | 轨迹优化 (TrajOpt)、MPC |

### 4.3 操作控制

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **gymnasium** | **1.0.0** | 标准 RL 环境接口 | 技能策略的 Gym API 封装 |

---

## 五、Phase 2 — 端到端任务执行（计划依赖）

### 5.1 VLM / LLM

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **transformers** | **4.51.0** | HuggingFace VLM/LLM 推理 | 场景理解、任务分解 |
| **accelerate** | **1.4.0** | 多 GPU / 混合精度推理加速 | transformers 配套 |
| **sentencepiece** | **0.2.0** | Tokenizer 支持 | LLaMA 等模型分词的底层依赖 |

### 5.2 视觉语言模型（可选方案）

| 模型/服务 | 接入方式 | 说明 |
|----------|---------|------|
| GPT-4V / GPT-4o | OpenAI API (`openai>=1.68.0`) | 云端推理，无需本地 GPU |
| Claude 4 (Sonnet/Opus) | Anthropic API (`anthropic>=0.49.0`) | 替代方案 |
| Qwen2.5-VL (本地) | transformers + `qwen-vl-utils>=0.0.8` | AutoDL RTX 5090 本地部署 |
| MiniCPM-V (轻量) | transformers | 资源受限时备选，显存 < 8 GB |

### 5.3 中介服务

| 包名 | 版本 | 用途 |
|------|------|------|
| **httpx** | **0.28.1** | 异步 HTTP 客户端，调用 LLM API |
| **tenacity** | **9.0.0** | API 调用重试机制 |

---

## 六、Phase 3 — 学习驱动策略（计划依赖）

### 6.1 仿真训练

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **dm_control** | **1.0.25** | DeepMind Control Suite，MuJoCo RL 环境 | 需 `mujoco>=3.2.0` |
| **stable-baselines3** | **2.5.0** | PPO / SAC 强化学习算法实现 | 需 `torch>=2.4`, `gymnasium>=1.0` |
| **tensorboard** | **2.18.0** | 训练曲线可视化 | sb3 日志依赖 |

### 6.2 Isaac Lab (Isaac Sim RL 框架)

| 组件 | 版本 | 安装方式 | 说明 |
|------|------|---------|------|
| Isaac Sim | **4.5.0** | NGC Docker (`nvcr.io/nvidia/isaac-sim:4.5.0`) | 基础仿真平台 |
| Isaac Lab | **2.1.0** | `pip install isaac-lab==2.1.0` | RL 训练框架，替代原 IsaacGym 脚本 |
| isaaclab-rl | **2.1.0** | Isaac Lab 内嵌 | PPO/SAC/Dreamer 实现 |

### 6.3 模仿学习

| 包名 | 版本 | 用途 |
|------|------|------|
| **h5py** | **3.12.1** | 演示数据存储 (HDF5 格式) |
| **diffusers** | **0.32.0** | 扩散策略 (Diffusion Policy) 推理 |

---

## 七、Phase 4 — 高级能力（计划依赖）

| 包名 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **trimesh** | **4.5.0** | 三角网格碰撞检测 | 灵巧操作中的接触建模 |
| **cvxopt** | **1.3.2** | 凸优化求解器 | Whole-Body MPC QP 求解 |
| **networkx** | **3.4** | 图结构算法 | 多机器人协作任务分配 |

---

## 八、ROS 2 中间件

| 组件 | 版本 | 安装方式 | 说明 |
|------|------|---------|------|
| ROS 2 | **Humble (LTS)** | `apt install ros-humble-*` | Ubuntu 22.04 官方 LTS，支持至 2027.05 |
| rosbridge | **2.1.5** | `apt install ros-humble-rosbridge-suite` | WebSocket ↔ ROS 2 桥接 |
| MoveIt 2 | **2.8.0** | `apt install ros-humble-moveit` | 运动规划框架（可选，可用自研替代） |

---

## 九、开发与代码质量工具

### 9.1 静态检查与格式化

| 工具 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **ruff** | **0.9.0** | Lint + 格式化（替代 flake8 + isort + black） | Rust 实现，速度快 10-100x |
| **mypy** | **1.14.0** | 静态类型检查 | `--ignore-missing-imports` 用于缺少 stubs 的第三方库 |
| **pre-commit** | **4.1.0** | Git hook 管理 | `.pre-commit-config.yaml` 自动触发 |

### 9.2 测试工具

| 工具 | 版本 | 用途 | 说明 |
|------|------|------|------|
| **pytest** | **8.3.4** | 测试框架 | 单元测试 + 集成测试 |
| **pytest-cov** | **6.0.0** | 代码覆盖率报告 | `--cov` + HTML 报告 |
| **pytest-xdist** | **3.6.1** | 并行测试执行 | `-n auto` 加速 CI |
| **pytest-timeout** | **2.3.3** | 测试超时保护 | 防止死循环测试挂死 CI |
| **pytest-benchmark** | **5.1.0** | 性能基准测试 | 性能回归检测 |

### 9.3 代码质量配置文件

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies: [numpy>=2.0]
```

---

## 十、AutoDL 云环境

### 10.1 实例配置

| 配置项 | 规格 |
|--------|------|
| GPU | NVIDIA RTX 5090 × 1 (32 GB GDDR7) |
| CPU | 16 vCPU (AMD EPYC / Intel Xeon) |
| 内存 | 64 GB DDR5 |
| 系统盘 | 100 GB SSD |
| 数据盘 | 200 GB SSD (挂载 `/root/autodl-tmp`) |
| 镜像 | Ubuntu 22.04 + CUDA 12.4.1 + Miniconda 24.11.0 |

### 10.2 Docker 容器

| 镜像 | 版本 | 用途 |
|------|------|------|
| `nvcr.io/nvidia/isaac-sim` | **4.5.0** | Isaac Sim headless + Streaming Client |
| `nvcr.io/nvidia/pytorch` | **25.02** | PyTorch 训练容器（可选替代） |

### 10.3 环境启动检查清单

```bash
# 1. GPU 可用性
nvidia-smi | grep "RTX 5090"

# 2. CUDA 版本
nvcc --version | grep "12.4"

# 3. Docker 状态
docker info > /dev/null 2>&1 && echo "Docker OK"

# 4. 数据盘挂载
df -h /root/autodl-tmp | tail -1

# 5. Isaac Sim 容器
docker ps --filter "name=isaac-sim" --format "{{.Status}}"

# 6. Python 版本
python3.12 --version
```

---

## 十一、完整依赖安装脚本

### 11.1 Phase 0（当前阶段）

```bash
# 创建虚拟环境
python3.12 -m venv .venv --prompt robot
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装核心依赖
pip install --upgrade pip setuptools wheel
pip install mujoco==3.2.2
pip install numpy==2.1.3
pip install opencv-python==4.10.0
pip install pynput==1.7.7

# 安装测试依赖
pip install pytest==8.3.4 pytest-cov==6.0.0

# 安装开发工具
pip install ruff==0.9.0 mypy==1.14.0 pre-commit==4.1.0
```

### 11.2 Phase 1-2（执行感知与操作）

```bash
# 感知
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install open3d==0.19.0

# 规划与控制
pip install scipy==1.15.0 pin==3.2.0 casadi==3.6.7 gymnasium==1.0.0

# VLM / LLM
pip install transformers==4.51.0 accelerate==1.4.0 sentencepiece==0.2.0
pip install httpx==0.28.1 tenacity==9.0.0

# API 客户端（按需选择）
pip install openai>=1.68.0       # GPT-4V
pip install anthropic>=0.49.0    # Claude
```

### 11.3 Phase 3（学习策略）

```bash
# RL 训练
pip install dm_control==1.0.25
pip install stable-baselines3==2.5.0 tensorboard==2.18.0

# 模仿学习
pip install h5py==3.12.1 diffusers==0.32.0

# Isaac Lab (需先启动 Isaac Sim 容器)
pip install isaac-lab==2.1.0
```

### 11.4 Phase 4（高级能力）

```bash
pip install trimesh==4.5.0 cvxopt==1.3.2 networkx==3.4
```

### 11.5 一键安装全部依赖

```bash
# 全部 pip 依赖（不含 Isaac Lab / ROS 2）
pip install \
  mujoco==3.2.2 \
  numpy==2.1.3 \
  opencv-python==4.10.0 \
  pynput==1.7.7 \
  torch==2.6.0 torchvision==0.21.0 \
  open3d==0.19.0 \
  scipy==1.15.0 pin==3.2.0 casadi==3.6.7 \
  gymnasium==1.0.0 dm_control==1.0.25 \
  stable-baselines3==2.5.0 tensorboard==2.18.0 \
  transformers==4.51.0 accelerate==1.4.0 sentencepiece==0.2.0 \
  h5py==3.12.1 diffusers==0.32.0 \
  httpx==0.28.1 tenacity==9.0.0 \
  trimesh==4.5.0 cvxopt==1.3.2 networkx==3.4 \
  pytest==8.3.4 pytest-cov==6.0.0 pytest-xdist==3.6.1 pytest-timeout==2.3.3 \
  ruff==0.9.0 mypy==1.14.0 pre-commit==4.1.0
```

---

## 十二、版本兼容性矩阵

### 12.1 Python 3.12 兼容性

| 包名 | Python 3.12 支持 | 说明 |
|------|:---:|------|
| mujoco 3.2.x | ✅ | 官方 wheel，cp312 支持 |
| numpy 2.1.x | ✅ | 完整 3.12 支持 |
| opencv-python 4.10.x | ✅ | cp312 wheel 可用 |
| pynput 1.7.x | ✅ | 纯 Python，无版本限制 |
| torch 2.6.0 | ✅ | 官方 CUDA 12.4 + cp312 wheel |
| open3d 0.19.x | ✅ | cp312 wheel 可用 |
| pin 3.2.x | ✅ | conda-forge/pip cp312 |
| casadi 3.6.x | ✅ | cp312 wheel 可用 |
| scipy 1.15.x | ✅ | 完整 3.12 支持 |
| gymnasium 1.0.x | ✅ | 纯 Python |
| transformers 4.51.x | ✅ | 纯 Python |
| stable-baselines3 2.5.x | ✅ | 纯 Python，torch 依赖链 |
| dm_control 1.0.x | ✅ | mujoco 依赖链，cp312 支持 |
| dm_control 1.0.x | ✅ | mujoco 依赖链，cp312 支持 |
| ruff 0.9.x | ✅ | Rust 编译，cp312 |
| mypy 1.14.x | ✅ | cp312 wheel |

### 12.2 CUDA 12.4 兼容性

| 包名 | CUDA 12.4 支持 | 说明 |
|------|:---:|------|
| torch 2.6.0 | ✅ | cu124 wheel 官方发布 |
| mujoco 3.2.2 | ✅ | GPU 加速渲染，CUDA 12.x 兼容 |
| open3d 0.19.0 | ✅ | CUDA 加速点云处理 |

### 12.3 关键依赖关系链

```
torch 2.6.0
├── torchvision 0.21.0 (需匹配 torch 主版本)
├── stable-baselines3 2.5.0 (需 torch >= 2.4)
└── transformers 4.51.0 (需 torch >= 2.1)

mujoco 3.2.2
└── dm_control 1.0.25 (需 mujoco >= 3.2.0)

gymnasium 1.0.0
└── stable-baselines3 2.5.0 (需 gymnasium >= 1.0)
```

---

## 十三、依赖安全性注意事项

| 风险项 | 建议 |
|--------|------|
| `torch` 官方索引 | 始终从 `https://download.pytorch.org/whl/cu124` 安装，不要用 PyPI 的 CPU-only 版本 |
| `numpy 2.x` 迁移 | 项目中使用了 `numpy` 基本操作，2.x 无破坏性变更影响当前代码。注意第三方包是否适配 numpy 2.x |
| `opencv-python` vs `opencv-python-headless` | 服务器环境（无 GUI）建议用 `opencv-python-headless==4.10.0` 减少 X11 依赖 |
| Isaac Sim Docker 授权 | NGC 镜像需接受 NVIDIA EULA，首次拉取需登录 `nvcr.io` |

---

> **更新日志**
> - 2026-05-29 (v1): 初始版本，基于 PROJECT.md / PROGRESS.md / TEST_PLAN.md 及源码导入分析，覆盖 Phase 0-4 全部依赖，明确 Python 3.12.9 及所有技术组件版本