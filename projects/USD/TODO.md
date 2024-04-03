# 基本
* [x] 配开发环境
* [x] 写一个基本的USD导入导出，先跑通整个流程
* [x] 研究USD应该如何设计融入zeno
* [x] 附带数据与类型 attributes可以直接输出成string 或者用一个结构体继承INode实现持久化
* [ ] 摸索各个平台和编译器的构建(linux cmake vs)+写文档 @hcx
* [ ] 考虑将USD封装为lib库 @hcx
# 编辑方案
* [x] 最终决定: 将USD prim用等价UI节点的方式eval出来
* [ ] 其他坑点有待研究
# 几何相关
* [x] 下面这些全部用UI吐节点的方式实现
* [x] 支持XformOp导入为链式操作
  * [x] 基本变换支持：位移、缩放、欧拉旋转
  * [x] 支持orient
  * [ ] 支持xformOp中的reset、revert操作
  * [ ] 支持xformOp中的Matrix信息
* [x] Cube导入
* [x] Sphere导入
* [x] Cylinder导入
* [x] Cone导入
* [x] Capsule导入
* [x] Plane导入
* [ ] 实现Mesh导入构建
  * [x] 基础的mesh导入: 吐一个ImportUSDMesh节点
  * [ ] 加一个mesh纹理支持
  * [ ] 支持骨骼 Skeleton导入，并用骨骼初始化mesh的transform
  * [ ] 支持动画 SkelAnimation导入 并且支持多帧播放
* [ ] 支持递归引入 例如import一个根节点 其下的叶节点也一并导入(sublayer??)
* [x] 支持编辑USD prim: 目前方案是导入时转换为等价的zeno节点树
* [ ] 支持导出USD prim
* [ ] prim可见性为inherited处理
# mtlx @hcx
* [ ] Material/mtlx
* [ ] Shader
# 灯光相机
* [x] CylinderLight
* [x] DiskLight
* [x] ~~DistantLight: 目前不支持无形光源 暂时忽略~~
* [x] DomeLight: 简单支持了一下
* [x] RectLight
* [x] SphereLight
* [ ] GeometryLight
* [ ] Camera
# instancing
* [ ] 实现实例化 USD instancing
# 进阶以及可选
* [ ] Scope?
* [ ] OpenVDBAsset?
* [ ] Volume?
* [ ] SkelRoot?
* [ ] xformop列表编辑: 已经支持了xform序列转换 还要支持导出
* [ ] Collapse功能|节点支持
* [ ] 其他USD合成操作
* [ ] Points?
* [ ] Curves?
