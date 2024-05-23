# 近期重要性排序
* 材质解析优化 primvar displayColor等
* 各种读取方式改接口化 而不是硬读attribute (xform这些)
* 各种做安全判断
* 各种节点都尝试改成按帧读取 Camera XformOps
---
* 导出方案
* import性能优化
* 子图生成与reference
* visible的树形解析
---
* 测一下镜头
* instancing
# 基本
* [x] 配开发环境
* [x] 写一个基本的USD导入导出，先跑通整个流程
* [x] 研究USD应该如何设计融入zeno
* [x] 附带数据与类型 attributes可以直接输出成string 或者用一个结构体继承INode实现持久化
* [x] 摸索各个平台和编译器的构建(linux cmake vs)+写文档 @hcx
* [x] 考虑将USD封装为lib库 @hcx
* [ ] 各种测试
# 编辑方案
* [x] 最终决定: 将USD prim等价转换为zeno节点系统 用户编辑节点即可实现USD编辑
* [x] 生成节点排列 在生成节点时根据层级关系排列节点
  * [x] 基本实现
  * [x] 有概率闪退/中断
  * [x] MakeList子节点乱序问题
* [ ] Reference与Sublayer的处理: zeno如何体现这两个操作
  * 其内容都解析到subgraph里面
  * 在subgraph节点后面加一个userdata 表示ref还是sublayer或者别的
* [ ] 交互优化：总不能让用户一个个选prim吧
# Xform
* [x] 支持把单个Xform的XformOp导入为链式操作
  * [x] 基本变换支持：位移、缩放、欧拉旋转
  * [x] 支持orient(四元数)
  * [x] 支持xformOp中的Matrix类型导入
  * [ ] 对于pivot操作的支持 目前暂时先考虑支持xformOp:translate:pivot这一个 其他的没什么资料可以查 可以支持导入，但是如何用zeno节点表示 `设置pivot` 这件事是个问题 否则没法导出
  * [x] 对于inverse操作的支持 !invert!
  * [ ] resetXformStack情况的处理 !resetXformStack! 不继承父节点的transform
  * [ ] translate、scale这些zeno节点没办法按照current frame来吐(?) 除非全部放到ImportUSDPrimMatrix里面去
* [x] transform树结构导入: 共用问题, zeno的不同geo节点没办法共用同一个transform节点: MakeList节点解决
# 树形解析
* [x] 基本解析框架
* [x] 使用MakeList构造树形层级
* [x] 使用SetUserData2标记prim信息
* [x] 节点排列算法
* [x] 对于同时包含xform和geom的根节点的处理方案
* [x] zeno的shaderFinalize可以单独一棵树 需要稍微重构一下树形排列的做法 支持多棵树 可以考虑把shader finalize放进subgraph里面
* [ ] visible的树形关系处理
* [ ] 同一个子图的多棵树生成时重叠
* [ ] 考虑共用节点的情况 例如GetFrameNum的输出可能会被很多节点同时作为input使用
# 几何相关
* [x] 下面这些全部用UI吐节点的方式实现
* [x] Cube导入
* [x] Sphere导入
* [x] Cylinder导入
* [x] Cone导入
* [x] Capsule导入
* [x] Plane导入
* [x] Mesh导入构建
  * [x] 基础的mesh导入: 吐一个ImportUSDMesh节点
  * [x] 解析material引用 并且材质会被放进subgraph中生效
  * [x] 支持 SkelRoot+Skeleton+SkelAnimation解析 并且支持多帧播放
  * [x] BlendShape解析支持
* [x] 支持递归导入prim树
* [x] 支持编辑USD prim: 目前方案是导入时转换为等价的zeno节点树
* [ ] 支持导出USD prim: 遍历node 然后利用userdata标记的信息反向导出 具体实现有待研究
  * 节点系统完全转译回USD可能会遇到信息不保真的问题
  * 如何获取用户修改了哪部分 仅保存修改的部分
# mtlx
* [ ] Material支持
  * [x] 基础解析功能
  * [x] mesh绑定映射自动化
    * 用UsdShadeMaterialBindingAPI::ComputeBoundMaterial搞定了
  * [x] 用一个专门的subgraph存储材质定义节点(shaderFinalize)
  * [ ] Shader input解析支持
    * [x] 字面值解析支持
    * [x] 纹理采样支持
    * [ ] primvar取值: 可以用ShaderInputAttr节点来表示
      * UsdPrimvarReader或者UsdPrimvarReader_float2用来获取uv坐标
      * displayColor解析
    * 其他input属性支持 找一下包含有所有属性的表格
  * [ ] SpecularColor 这个属性在zeno里面暂时没有支持 MetalColor也没有启用 综合来说这是一个不那么Physical的东西
  * [ ] mtlx
# 灯光相机
* [x] CylinderLight
* [x] DiskLight
* [x] ~~DistantLight: 目前不支持无形光源 暂时忽略~~
* [x] DomeLight: 简单支持了一下
* [x] RectLight
* [x] SphereLight
* [x] light texture支持: DomeLight RectLight
* [x] Camera: 简单支持了一下 暂无fov
  * [x] 基本支持
  * [ ] 动画？
  * [ ] 支持多帧读取
* [ ] 灯光动画？
# 优化
/Kitchen_set/Arch_grp/Kitchen_1
这个路径import进来有2467个节点 尽量尝试减少它的解析后节点数
直接import整个kitchen会卡很久然后炸
* [ ] 节点缩减
  * [x] 把setuserdata合并成一个点 可以减少解析后的节点数和层数(减少了503个)
  * [ ] 考虑支持某些节点共用 例如GetFrameNum可以全图共用
* [ ] 子图划分
  * [ ] 考虑做一下子图划分 防止解析出来的节点过多时 编辑界面卡顿+崩溃 但是哪些分成子图有待商榷
* [ ] 代码整理
  * [ ] usdnode代码太长 需要做一下划分，用不同的.cpp根据功能分开实现.h里面的内容 比如说geom的单开一个cpp 材质单开一个 xform单开一个 主流程一个等等
* [ ] 节点效率优化
  * [ ] 需要profile并关注一下性能瓶颈 看下哪里比较耗时 个人感觉是创建节点、节点编辑比较卡
* [ ] 考虑用异步实现，而不是长时间阻塞 或者考虑提升并行度、增加线程等
# instancing
* [ ] 实现实例化 USD instancing: 目前zeno对于instancing的支持有限 可以暂时往后稍稍?
# 其他属性
* [ ] Scope?
* [ ] OpenVDBAsset?
* [ ] Volume?
* [ ] SkelRoot?
* [ ] Points?
* [ ] Curves?
# 进阶以及可选
* [ ] Collapse功能 | 节点支持
* [ ] 支持其他USD合成操作表示
