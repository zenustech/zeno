# 基本
* [x] 配开发环境
* [x] 写一个基本的USD导入导出，先跑通整个流程
* [x] 研究USD应该如何设计融入zeno
* [x] 附带数据与类型 attributes可以直接输出成string 或者用一个结构体继承INode实现持久化
* [ ] 摸索各个平台和编译器的构建(linux cmake vs)+写文档
# 编辑方案
* [x] 将USD prim导入为zeno prim 就可以直接用zeno的编辑系统进行编辑
* [ ] 其他坑点有待研究
# 几何相关
* [x] 实现Mesh导入构建
* [x] 支持Xform 目前支持将xformop list合成单个matrix导入zeno
* [ ] 支持骨骼 Skeleton导入，并用骨骼初始化mesh的transform
* [ ] 支持动画 SkelAnimation导入 并且支持多帧播放
* [ ] 支持层级引入 例如import一个根节点 其下的叶节点也一并导入(sublayer??)
* [ ] 支持各种基础几何的导入构建 Sphere Cube Cylinder Point Line等
* [ ] 实现以上和以下所有内容的导出方案(如果能导入 一般来说导出就没什么问题了)
# mtlx
* [ ] 支持Material/mtlx + Shader
# 灯光相机
* [ ] 实现灯光相机 各种xxxLight Camera
# instancing
* [ ] 实现实例化 USD instancing
# 进阶以及可选
* [ ] xformop列表编辑
* [ ] Collapse功能|节点支持
* [ ] 合成操作
