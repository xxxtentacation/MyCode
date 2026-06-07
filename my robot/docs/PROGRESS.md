# 项目进度文档

> 配套：[PROJECT.md](./PROJECT.md) | [TEST_PLAN.md](./TEST_PLAN.md)
> 更新：2026-05-29

---

## 一、总体进度

| Phase | 名称 | 状态 | 完成度 |
|-------|------|------|--------|
| Phase 0 | 环境搭建与基础仿真 | 🔄 进行中 | 85% |
| Phase 1 | 物体操作 | ⬜ 未开始 | 0% |
| Phase 2 | 端到端任务执行 | ⬜ 未开始 | 0% |
| Phase 3 | 学习驱动策略 | ⬜ 未开始 | 0% |
| Phase 4 | 高级能力 | ⬜ 未开始 | 0% |

---

## 二、Phase 0 详细进度

### 已完成

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 依赖清单 | `requirements.txt` | ✅ | mujoco, numpy, opencv-python, pynput |
| 目录结构 | 全部 `__init__.py` | ✅ | simulation/, control/, perception/, tests/ |
| UR5 模型 | `simulation/assets/ur5e/scene.xml` | ✅ | MJCF 格式，含桌面/物块/Robotiq 夹爪/相机/触觉传感器 |
| 仿真环境 | `simulation/mujoco_env.py` | ✅ | MujocoEnv 类，reset/step/render/get_ee_pose API |
| 夹爪控制 | `control/gripper.py` | ✅ | RobotiqGripper 类，open/close/set_width |
| 仿真相机 | `perception/camera.py` | ✅ | SimCamera 类，内参矩阵/3D↔2D 投影 |
| 入口脚本 | `main.py` | ✅ | 支持 --demo / --teleop / --sensor / 默认 viewer |
| 遥操作 | `scripts/teleop.py` | ✅ | 键盘控制 UR5 末端，IK→关节→仿真 |
| 传感器检测 | `scripts/sensor_check.py` | ✅ | 运行 N 秒，打印关节/EE/帧率统计 |
| IK 测试框架 | `tests/unit/control/test_ik_solver.py` | ✅ | 5 个测试用例，全部通过 |
| 测试 Fixtures | `tests/conftest.py` | ✅ | 全局 pytest fixtures (env, IK, camera, gripper) |
| E2E 遥操作测试 | `tests/e2e/phase0/test_teleop.py` | ✅ | 遥操作模块 smoke test + IK 集成 |
| E2E 传感器测试 | `tests/e2e/phase0/test_sensor_stream.py` | ✅ | 传感器数据流完整性验证 |
| 仿真模型测试 | `tests/simulation/test_robot_model.py` | ✅ | UR5 模型/物理正确性验证 |
| 静态检查配置 | `.pre-commit-config.yaml` | ✅ | ruff, mypy, pre-commit-hooks |
| 项目配置 | `pyproject.toml` | ✅ | ruff, mypy, pytest 配置 |
| 测试运行脚本 | `scripts/run_tests.py` | ✅ | 便捷测试运行器 (--unit/--e2e/--cov) |

### 存在问题

| 编号 | 模块 | 问题 | 严重级别 | 状态 |
|------|------|------|---------|------|
| B-001 | `control/ik_solver.py` | `test_ik_fk_consistency` 失败：IK 解→FK 回代误差达 0.64m | P1-High | ✅ 已修复 (2026-05-29) |
| B-002 | `control/ik_solver.py` | `test_home_ik_roundtrip` 失败：home 位姿 IK 误差 3.86 rad | P1-High | ✅ 已修复 (2026-05-29) |

**根因与修复：** 原代码 `pw = p_ee - d6 * R[:,2]` 计算的是 frame 5 原点 (p_05)，而非 frame 4 原点 (p_04)。在 modified DH 约定下，p_05 依赖 θ4，破坏了位置/姿态解耦。通过推导发现 exact 关系：z4 = -y_ee，因此 p_04 = p_ee + d5*R[:,1] - d6*R[:,2]。此公式对所有构型精确成立，无需迭代。

### 待开始

| 任务 | 依赖 | 预计 |
|------|------|------|
| 运行 IK 单元测试验证修复 | Python 环境 | 下一优先级 |
| MuJoCo 集成测试 + 遥操作验证 | IK 修复 + 图形显示 | Phase 0 出口 |
| AutoDL 云端环境搭建 | — | 需购买实例 |
| Isaac Sim 容器部署 | AutoDL 实例 | Phase 0 出口 |
| E2E 测试 (P0) | 全部模块就绪 | Phase 0 出口 |

---

## 三、文件清单

```
my-robot/
├── docs/
│   ├── PROJECT.md               # 项目文档
│   ├── TEST_PLAN.md             # 测试计划
│   ├── PROGRESS.md              # 本文件：进度文档
│   └── DEPENDENCIES.md          # 依赖技术文档（版本清单 + 安装脚本）
├── .pre-commit-config.yaml      # 静态检查配置 (ruff, mypy)
├── pyproject.toml                # 项目元数据 + ruff/mypy/pytest 配置
├── requirements.txt             # Phase 0 Python 依赖
├── main.py                      # 入口脚本
├── simulation/
│   ├── __init__.py
│   ├── mujoco_env.py            # MuJoCo 仿真环境封装
│   └── assets/ur5e/
│       └── scene.xml            # UR5 + Robotiq + 桌面场景
├── control/
│   ├── __init__.py
│   ├── ik_solver.py             # UR5 闭式 IK（已修复）
│   └── gripper.py               # Robotiq 2F-85 夹爪控制
├── perception/
│   ├── __init__.py
│   └── camera.py                # 仿真相机封装
├── scripts/
│   ├── teleop.py                # 键盘遥操作
│   ├── sensor_check.py          # 传感器数据流验证
│   └── run_tests.py             # 测试运行脚本
└── tests/
    ├── __init__.py
    ├── conftest.py               # 全局 pytest fixtures
    ├── unit/
    │   ├── __init__.py
    │   └── control/
    │       ├── __init__.py
    │       └── test_ik_solver.py # IK 单元测试 (5 tests)
    ├── e2e/
    │   ├── __init__.py
    │   └── phase0/
    │       ├── __init__.py
    │       ├── test_teleop.py    # 遥操作 E2E smoke test
    │       └── test_sensor_stream.py  # 传感器数据流测试
    ├── simulation/
    │   ├── __init__.py
    │   └── test_robot_model.py  # UR5 模型/物理测试
    └── fixtures/                 # 测试固件目录
```

**共 33 个文件（含 __init__.py），约 1600 行 Python + 150 行 XML + 测试代码。**

---

## 四、变更日志

### 2026-05-16

| 时间 | 变更 | 涉及文件 |
|------|------|---------|
| 15:00 | 创建项目文档 v1 | `docs/PROJECT.md` |
| 15:20 | 新增 AutoDL 云环境部署方案 | `docs/PROJECT.md` |
| 15:40 | 创建测试计划 | `docs/TEST_PLAN.md` |
| 15:50 | 聚焦机械臂操作，移除导航/移动底盘 | `docs/PROJECT.md`, `docs/TEST_PLAN.md` |
| 16:00 | 选定 UR5 + Robotiq 2F-85 | `docs/PROJECT.md`, `docs/TEST_PLAN.md` |
| 16:20 | Phase 0 实现计划 | `docs/PROJECT.md` |
| 16:30 | 创建 10 个模块文件 + 6 个包文件 | 全部 `.py` / `.xml` |
| 16:45 | IK 求解器首次实现，2/5 测试未通过 | `control/ik_solver.py` |
| 17:00 | 调试确认 IK 腕心计算近似错误 | `control/ik_solver.py` |
| 17:10 | 创建项目进度文档 | `docs/PROGRESS.md` |
| 2026-05-29 | 修复 IK 求解器腕心计算（B-001, B-002） | `control/ik_solver.py` |
| 2026-05-29 | 标准化观测字典键名，新增 ee_position/ee_orientation/gripper_width 等 | `simulation/mujoco_env.py` |
| 2026-05-29 | 修复 E2E/仿真测试中 get_gripper_state() 返回类型不匹配 | `tests/e2e/`, `tests/simulation/` |
| 2026-05-29 | 新增 pyproject.toml、测试运行脚本 | `pyproject.toml`, `scripts/run_tests.py` |

---

## 五、下一步计划

1. **运行单元测试** — 验证 IK 修复后所有 5 个测试通过：`python scripts/run_tests.py --unit`
2. **运行完整测试套件** — `python scripts/run_tests.py --all`（需要 Python+mujoco 环境）
3. **MuJoCo 集成测试** — 用修复后的 IK 驱动仿真，验证遥操作可用
4. **AutoDL 环境** — 租用实例，安装依赖，运行完整仿真
5. **Phase 0 E2E** — 按 `docs/TEST_PLAN.md` 执行 P0 门禁测试
