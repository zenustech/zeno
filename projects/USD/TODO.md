# 近期重要性排序
* [ ] instancing
* [ ] 材质shader
* [ ] 导出
* [ ] mesh纹理
* [ ] 骨骼+动画

# 基本
* [x] 配开发环境
* [x] 写一个基本的USD导入导出，先跑通整个流程
* [x] 研究USD应该如何设计融入zeno
* [x] 附带数据与类型 attributes可以直接输出成string 或者用一个结构体继承INode实现持久化
* [ ] 摸索各个平台和编译器的构建(linux cmake vs)+写文档 @hcx
* [x] 考虑将USD封装为lib库 @hcx
* [ ] 容暴性测试
# 编辑方案
* [x] 最终决定: 将USD prim等价转换为zeno节点系统 用户编辑节点即可实现USD编辑
* [x] 生成节点排列 在生成节点时根据层级关系排列节点
# 几何相关
* [x] 下面这些全部用UI吐节点的方式实现
* [x] 支持XformOp导入为链式操作
  * [x] 基本变换支持：位移、缩放、欧拉旋转
  * [x] 支持orient
  * [x] 支持xformOp中的Matrix类型导入
  * [ ] 支持xformOp中的reset、revert操作??
* [x] Cube导入
* [x] Sphere导入
* [x] Cylinder导入
* [x] Cone导入
* [x] Capsule导入
* [x] Plane导入
* [ ] 实现Mesh导入构建
  * [x] 基础的mesh导入: 吐一个ImportUSDMesh节点
  * [ ] mesh纹理支持
  * [ ] 支持骨骼 Skeleton导入，并用骨骼初始化mesh的transform
  * [ ] 支持动画 SkelAnimation导入 并且支持多帧播放
* [x] 支持递归引入 例如import一个根节点 其下的叶节点也一并导入
* [x] 支持编辑USD prim: 目前方案是导入时转换为等价的zeno节点树
* [ ] 支持导出USD prim
* [ ] prim可见性为inherited的处理
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
* [x] light texture支持: DomeLight RectLight
* [x] Camera: 简单支持了一下 暂无fov
# instancing
* [ ] 实现实例化 USD instancing
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
