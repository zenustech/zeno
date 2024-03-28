# 基本
* [x] 配开发环境
* [x] 写一个基本的USD导入导出，先跑通整个流程
* [x] 研究USD应该如何设计融入zeno
* [x] 附带数据与类型 attributes可以直接输出成string 或者用一个结构体继承INode实现持久化
* [ ] 摸索各个平台和编译器的构建(linux cmake vs)+写文档
* [ ] 考虑将USD封装为lib库
# 编辑方案
* [x] 将USD prim导入为zeno prim 就可以直接用zeno的编辑系统进行编辑
* [ ] 其他坑点有待研究
# 几何相关
* [x] 实现Mesh导入构建
* [x] 支持Xform 目前支持将xformop list合成单个matrix导入zeno
* [x] Cube基本导入
* [x] Sphere导入(没有规定精细度)
* [x] Cylinder导入
* [x] Cone导入
* [ ] Capsule 差一个对axis的支持
* [x] Plane
* [ ] 支持层级引入 例如import一个根节点 其下的叶节点也一并导入(sublayer??)
* [ ] 支持骨骼 Skeleton导入，并用骨骼初始化mesh的transform
* [ ] 支持动画 SkelAnimation导入 并且支持多帧播放
* [ ] 如何编辑USD prim
* [ ] 实现所有已支持内容的对应导出方案
# mtlx @hcx
* [ ] Material/mtlx
* [ ] Shader
# 灯光相机 @hcx
* [x] CylinderLight
* [x] DiskLight
* [ ] DistantLight: 目前不支持无形光源 暂时忽略
* [ ] DomeLight: 暂时没有支持texture
* [ ] RectLight
* [ ] SphereLight
* [ ] GeometryLight
* [ ] Camera
# instancing
* [ ] 实现实例化 USD instancing
# 进阶以及可选
* [ ] xformop列表编辑: 目前会对XformOp拍成一个矩阵来输出 要找个办法支持编辑
* [ ] Collapse功能|节点支持
* [ ] 其他USD合成操作
* [ ] Points?
* [ ] Curves?
