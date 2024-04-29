# 近期重要性排序
* [ ] 材质shader解析
* [ ] 骨骼+动画
* [ ] 支持帧
* [ ] 导出
* [ ] BlendShape
* [ ] import性能优化
* [ ] 子图生成

* [ ] instancing

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
* [ ] Reference与sublayer的处理
* [ ] 交互优化：总不能让用户一个个选prim吧
# Xform
* [x] 支持把单个Xform的XformOp导入为链式操作
  * [x] 基本变换支持：位移、缩放、欧拉旋转
  * [x] 支持orient(四元数)
  * [x] 支持xformOp中的Matrix类型导入
  * [ ] 对于pivot操作的支持 目前暂时先考虑支持xformOp:translate:pivot这一个 其他的没什么资料可以查 可以支持导入，但是如何用zeno节点表示 `设置pivot` 这件事是个问题 否则没法导出
/Kitchen_set/Arch_grp/Kitchen_1/Geom/Cabinets/Body/pCube251
  * [x] 对于inverse操作的支持 !invert!
  * [ ] resetXformStack情况的处理 !resetXformStack! 不继承父节点的transform
这尼玛也太坑爹了 方案就是在整条路径上专门剔除掉这个prim的变换 或者不放进makelist里面，然后tag写它的完整路径而不是name
* [x] transform树结构导入: 共用问题, zeno的不同geo节点没办法共用同一个transform节点: MakeList节点解决
/Kitchen_set/Props_grp/North_grp/SinkArea_grp
# 树形生成
* [x] 基本生成框架
* [x] 使用MakeList构造树形层级
* [x] 使用SetUserData2标记prim信息
* [x] 节点摆放算法
* [x] 对于同时包含xform和geom的根节点的处理方案
* [ ] zeno的shaderFinalize可以单独一棵树 需要稍微重构一下树形排列的做法 支持多棵树
# 几何相关
* [x] 下面这些全部用UI吐节点的方式实现
* [x] Cube导入
* [x] Sphere导入
* [x] Cylinder导入
* [x] Cone导入
* [x] Capsule导入
* [x] Plane导入
* [ ] 实现Mesh导入构建
/Kitchen_set/Arch_grp/Kitchen_1/Geom/Cabinets
  * [x] 基础的mesh导入: 吐一个ImportUSDMesh节点
  * [ ] 解析material引用
  * [ ] 支持骨骼 Skeleton导入，并用骨骼初始化mesh的transform
  * [ ] 支持动画 SkelAnimation导入 并且支持多帧播放
  * [ ] BlendShape支持
/HumanFemale_Group/HumanFemale/Geom/Face/Mouth/UpperMouth/UpperTeeth/LUpperTooth3_sbdv
* [x] 支持递归导入prim树
* [x] 支持编辑USD prim: 目前方案是导入时转换为等价的zeno节点树
* [ ] 支持导出USD prim: 遍历node 然后利用userdata标记的信息反向导出 具体实现有待研究
  * 节点系统完全转译回USD可能会遇到信息不保真的问题
* [ ] prim可见性为inherited的处理
# mtlx
* [ ] Material支持
  * [x] 基础解析功能
  * [ ] 全局材质表和mesh绑定映射自动化
  * [ ] 对于primvar类型input的处理: 可以用ShaderInputAttr来做
  * [ ] 注意：SpecularColor在zeno里面暂时没有 MetalColor也没有启用
* [ ] mtlx

/HumanFemale_Group/KidThinButtonDown/Looks/KidThinButtonDownMat
* [ ] mesh中的material:binding属性会写使用的material的prim path
  * /HumanFemale_Group/KidThinButtonDown/Looks/KidThinButtonDownMat
  * /HumanFemale_Group/KidThinButtonDown/Geom/Render/ButtonDownRenderMesh_sbdv
  * /HumanFemale_Group/HumanFemale/Geom/Body/Body_sbdv
  * /HumanFemale_Group/HumanFemale/Looks/HumanFemaleMat
* [ ] 对应mesh
  * /HumanFemale_Group/HumanFemale/Geom/Face/Eyes/REye/Sclera_sbdv
  * /HumanFemale_Group/HumanFemale/Geom/Face/Eyes/REye/Iris_sbdv
  * /HumanFemale_Group/HumanFemale/Geom/Face/Eyes/REye/Cornea_sbdv
* [ ] Shader
  * UsdPreviewSurface用来着色 关键属性 info:implementationSource info:id inputs:diffuseColor inputs:roughness inputs:specularColor inputs:useSpecularWorkflow
  * UsdUVTexturenode用来采样
  * UsdPrimvarReader或者UsdPrimvarReader_float2用来获取uv坐标
  * /HumanFemale_Group/HumanFemale/Looks/ScleraMat/EyeSurf
  * /HumanFemale_Group/HumanFemale/Looks/NailMat/NailSurf
  * /HumanFemale_Group/HumanFemale/Looks/IrisMat/EyeSurf
  * /HumanFemale_Group/HumanFemale/Looks/PupilMat/DisplayColorPrimvar
  * /HumanFemale_Group/HumanFemale/Looks/CorneaMat/EyeSurf
  * /HumanFemale_Group/HumanFemale/Looks/HumanFemaleMat/HumanFemaleSurf
  * /HumanFemale_Group/HumanFemale/Looks/HumanFemaleMat/diffuseColorTexture
# 灯光相机
* [x] CylinderLight
* [x] DiskLight
* [x] ~~DistantLight: 目前不支持无形光源 暂时忽略~~
* [x] DomeLight: 简单支持了一下
* [x] RectLight
* [x] SphereLight
* [x] light texture支持: DomeLight RectLight
* [x] Camera: 简单支持了一下 暂无fov
# 优化 1 2 2 2 2
/Kitchen_set/Arch_grp/Kitchen_1
这个路径import进来有2467个节点 尽量尝试减少它的解析后节点数
直接import整个kitchen会卡很久然后炸
* [x] 把setuserdata合并成一个点 可以减少解析后的节点数和层数(减少到503个)
* [ ] 需要profile并关注一下性能瓶颈 看下哪里比较耗时 个人感觉是创建节点、节点编辑比较卡
* [ ] 考虑用异步实现，而不是长时间阻塞 或者考虑提升并行度、增加线程等
* [ ] 考虑做一下子图划分 防止解析出来的节点过多时 编辑界面卡顿+崩溃
# instancing
* [ ] 实现实例化 USD instancing: 目前zeno对于instancing的支持有限 可以暂时往后稍稍
# 其他属性
* [ ] Scope?
* [ ] OpenVDBAsset?
* [ ] Volume?
* [ ] SkelRoot?
* [ ] Points?
* [ ] Curves?
# 进阶以及可选
* [ ] Collapse功能|节点支持
* [ ] 支持其他USD合成操作
