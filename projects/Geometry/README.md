## QuadMesh节点参数说明
| 参数 | 类型 | 说明 |
| --- | ---- | --- |
| prim | PrimitiveObject | 输入三角形网格。 |
| deterministic | bool | 使用较慢的确定性算法。 |
| crease | float | 判断为折痕的二面角阈值。 |
| smooth_iter | int | 平滑和光线追踪重投影的次数（默认2）。 |
| quad_dominant | bool | 生成以四边形为主的三角形-四边形混合网格。 |
| intrinsic | bool | 使用intrinsic方法（默认extrinsic方法）。 |
| boundary | bool | 对于非闭合的表面，保持边缘轮廓。（有保留边的输入时自动开启） |
| scale | float | 期望的世界坐标下边长。（与vert_num, face_num至多同时指定一项） |
| vert_num | int | 期望点数。 |
| face_num | int | 期望面数。 |
| line_pick_tag | string | 保留的边的属性名称，属性应为int类型，该属性值为1的边被保留。只能保留边方向，不能严格固定顶点位置。 |
| marked_lines | list of vec2i | 保留的边的列表。只能保留边方向，不能严格固定顶点位置。 |


## fTetWild节点参数说明
| 参数 | 类型 | 说明 |
| --- | ---- | --- |
| prim | PrimitiveObject | 输入三角形网格。 |
| input_dir | readpath | 输入三角形网格文件路径。如果有prim输入则忽略此项。 |
| output_dir | string | 输出四面体网格文件路径。若为空，当以prim输入时则不输出任何文件，以文件读入时则输出到“<原路经>/<原文件名>_.msh” |
| tag | readpath | 输入用于布尔运算的面片标记的文件路径。 |
| operation | enum | 布尔运算种类 |
| edge_length | float | 期望相对边长。默认0.05。 |
| epsilon | float | 期望相对偏差。默认1e-3。 |
| stop_energy | float | 停止优化的能量阈值。默认10。 |
| skip_simplify | bool | 跳过预处理。 |
| no_binary | bool | 以ascii格式输出。（仅在有输出文件时生效） |
| no_color | bool | 不输出颜色。（仅在有输出文件时生效） |
| smooth_open_boundary | bool | 对于非闭合表面，光滑其边缘。 |
| export_raw | bool | 输出原始结果。 |
| manifold_surface | bool | 处理输出为流形。 |
| coarsen | bool | 尽可能使输出稀疏。 |
| csg | readpath | 包含csg树的json文件路径。 |
| disable_filtering | bool | 不过滤掉外部的元素。 |
| use_floodfill | bool | 使用泛洪法提取内部体积。 |
| use_general_wn | bool | 使用通常的绕数。 |
| use_input_for_wn | bool | 使用输入表面得到绕数。 |
| bg_mesh | readpath | 用于得到长度场的背景网格（.msh格式）文件路径 |