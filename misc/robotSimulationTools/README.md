# Robot Simulation Tools

把 **Bullet / MuJoCo** 物理引擎的模型仿真结果，导出为 ISF 场景：静态结构（mesh/材质/相机/灯光/坐标变换）+ 动态状态（逐帧位姿/关节角）。核心层为纯内存构建，既可直接落盘，也可在 zeno 中不经写盘中转直接取用。

## 项目结构

```
robotSimulationTools/
├── bulletDemo/              # Bullet 模型数据（URDF + mesh/贴图，由压缩包解压得到）
├── mujocoDemo/              # MuJoCo 模型数据（MJCF XML + 资源文件，由压缩包解压得到）
├── scripts/
│   ├── bulletModels.py      # Bullet 内置模型清单 MODELS（模型名 -> URDF 相对路径、帧数、控制方式）
│   ├── mujocoModels.py      # MuJoCo 内置模型清单 MODELS
│   ├── exporter/            # ★ 核心导出器（静态 + 动态，与引擎无关）
│   │   ├── exporterStaticBullet.py
│   │   ├── exporterDynamicBullet.py
│   │   ├── exporterStaticMujoco.py
│   │   ├── exporterDynamicMujoco.py
│   │   └── policies/        #  动态仿真的关节控制策略（humanoid_wave 等）
│   ├── exportTodisk/        # 写盘式：跑仿真并输出 ISF JSON + 资源文件
│   ├── visualization/       # 独立可视化调试脚本
│   └── zenoExport/          # zeno 导入节点（纯内存，不经写盘）
```

统一的模型入口在 `scripts/bulletModels.py` 与 `scripts/mujocoModels.py` 的 `MODELS` 清单（目前各 5 个示例），脚本与 zeno 节点的模型定位、帧数、控制方式都从这里读取。

## 核心导出器（`scripts/exporter/`）

- **静态导出器** `ISFExporter*.build()`：解析模型，生成场景静态结构 —— 网格几何、材质/纹理、相机、灯光、坐标变换。**纯内存**返回 `{"isf": ..., "assets": {相对路径: bytes}}`，不写盘。
- **动态导出器** `DynamicSimulation.run()`：连接物理客户端、加载模型、初始化电机、按 `frames` 逐帧推进并采集每帧的位姿/关节角，返回逐帧动态数据。控制由 `policies/` 下的策略驱动（`sin`=正弦关节驱动，`zero`=自由运动）。

## 两种运行方式

**① 写盘（`scripts/exportTodisk/`）** —— 独立脚本，适合命令行调用/批量：

```bash
python scripts/exportTodisk/bulletStaticTodisk.py  --model kuka_iiwa --outdir out/bullet
python scripts/exportTodisk/bulletDynamicTodisk.py  --model kuka_iiwa --outdir out/bullet
python scripts/exportTodisk/mujocoStaticTodisk.py   --model humanoid --outdir out/mujoco
python scripts/exportTodisk/mujocoDynamicTodisk.py  --model humanoid --outdir out/mujoco
```

输出 `scene_static.isf.json`（静态）与 `scene_dynamic.isf.json`（动态），网格/贴图资源写在同一输出目录。批量导出见 `mujocoBatchExport.py` / `bulletBatchExport.py`。

**② 纯内存（`scripts/zenoExport/zenoBullet.py` / `zenoMujoco.py`）** —— 在 zeno 的 `SequentialPythonNode` 中运行，加载模型→仿真→把静态/动态结果直接写入链命名空间 `robotStatic` / `robotDynamic`，供 `ReadRobotSimluation` 使用，**全程不落盘中转**。模型名、帧数、控制方式、是否加地面等通过 `zeArgs` 传入（`path` 可选，缺省时自动定位到 `scripts/exporter`）。

## 模型示例压缩包（重要说明）

示例数据以压缩包形式提供（如根目录的 `bulletMujocoDemos.zip`）。请**解压到仓库根目录**，使包内顶层 `bulletDemo/`、`mujocoDemo/` 两个文件夹直接落在项目根下，例如：

```
robotSimulationTools/
├── bulletDemo/data/...      # ← 压缩包解压后
├── mujocoDemo/model/...     # ← 压缩包解压后
└── scripts/
```

这样 `bulletModels.py` / `mujocoModels.py` 里 `MODELS` 的相对路径（如 `bulletDemo/data/r2d2.urdf`、`mujocoDemo/model/humanoid/humanoid.xml`）才能与模型文件匹配。若解压后出现多余的中间层（如 `robotSimulationTools/bulletMujocoDemos/bulletDemo/`），把内层 `bulletDemo`/`mujocoDemo` 移动到项目根目录即可。