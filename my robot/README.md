# 具身智能仿真机器人

UR5 机械臂 + Robotiq 2F-85 夹爪，在 MuJoCo 物理引擎上的具身智能仿真系统。

## 快速开始

```bash
pip install -r requirements.txt
python main.py --demo     # 无头模式下运行 Demo 轨迹
python main.py --teleop   # 键盘遥操作（需图形显示）
python main.py --sensor   # 传感器数据流检查
```

## 运行测试

```bash
python scripts/run_tests.py --unit    # 单元测试
python scripts/run_tests.py --all     # 全部测试
```

## 项目文档

| 文档 | 说明 |
|------|------|
| [PROJECT.md](docs/PROJECT.md) | 项目架构、技术路线、Phase 规划 |
| [TEST_PLAN.md](docs/TEST_PLAN.md) | 测试策略、用例规划 |
| [PROGRESS.md](docs/PROGRESS.md) | 进度追踪、Bug 清单、变更日志 |
| [DEPENDENCIES.md](docs/DEPENDENCIES.md) | 依赖版本、安装脚本、兼容性矩阵 |
