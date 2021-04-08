'''
Zensim Node System Blender Intergration

Copyright (c) archibate <1931127624@qq.com> (2020- ). All Rights Reserved.
'''

node_descriptors = '''
EndFrame:(SRC)(DST)()(visualize)
FLIP_Solid_Modifie:(Particles,DynaSolid_SDF,StatSolid_SDF,SRC)(DST)()(FLIPSolver)
GetVDBPoints:(grid,SRC)(pars,DST)()(openvdb)
MakeMatrix:(SRC)(matrix,DST)(float3:position:0 0 0,float3:rotation:0 0 0,float3:scale:0 0 0)(misc)
MeshToSDF:(mesh,SRC)(sdf,DST)(float:voxel_size:0.08)(openvdb)
P2G_Advector:(Particles,Velocity,PostAdvVelocity,SRC)(DST)(float:time_step:0.04 0.0,float:dx:0.01 0.0,int:RK_ORDER:1 1 4,float:pic_smoothness:0.02 0.0 1.0)(FLIPSolver)
RandomParticles:(SRC)(pars,DST)(int:count:)(particles)
ReadObjMesh:(SRC)(mesh,DST)(string:path:)(trimesh)
ReadParticles:(SRC)(pars,DST)(string:path:)(particles)
ReadMesh:(SRC)(data,DST)(string:path:,string:type:float)(openvdb)
SetVDBTransform:(grid,SRC)(DST)(float:dx:0.08,float3:position:0 0 0,float3:rotation:0 0 0,float3:scale:1 1 1)(openvdb)
SimpleSolver:(ini_pars,SRC)(pars,DST)(float:dt:0.04,float3:G:0 0 1)(particles)
SleepMilis:(SRC)(DST)(int:ms:1000)(misc)
TransformMesh:(mesh,matrix,SRC)(mesh,DST)()(trimesh)
ViewMesh:(mesh,SRC)(DST)()(visualize)
ViewParticles:(pars,SRC)(DST)()(visualize)
WriteObjMesh:(mesh,SRC)(DST)(string:path:)(trimesh)
WriteParticles:(pars,SRC)(DST)(string:path:)(particles)
WriteMesh:(data,SRC)(DST)(string:path:)(openvdb)
'''

import bpy
from bpy.types import NodeTree, Node, NodeSocket
from bpy.utils import register_class, unregister_class

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem


class ZensimTree(NodeTree):
    '''Zensim node system for physics simulation'''
    bl_idname = 'ZensimTreeType'
    bl_label = "Zensim Node Tree"
    bl_icon = 'PHYSICS'


class ZensimTreeNode(Node):
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'ZensimTreeType'


class ZensimSocket(NodeSocket):
    '''Zensim socket'''
    bl_idname = 'ZensimSocketType'
    bl_label = "Socket"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (1.0, 0.4, 0.216, 1.0)


class ZensimNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'ZensimTreeType'


node_categories = [
    ZensimNodeCategory('TRIMESH', "trimesh", items=[
        NodeItem("ZensimNodeType_ReadObjMesh"),
    ]),
]


core_classes = [
    ZensimTree,
    ZensimSocket,
]

user_classes = []


def add_zensim_node_class(n_name, n_params, n_inputs, n_outputs):
    class Def(ZensimTreeNode):
        __doc__ = 'Zensim node: ' + n_name
        bl_idname = 'ZensimNodeType_' + n_name
        bl_label = n_name
        bl_icon = 'PHYSICS'

        def init(self, context):
            for name in n_inputs:
                self.inputs.new('ZensimSocketType', name)
            for name in n_outputs:
                self.outputs.new('ZensimSocketType', name)

        def draw_buttons(self, context, layout):
            for name, _, _ in n_params:
                layout.prop(self, name)

        def draw_buttons_ext(self, context, layout):
            for name, _, _ in n_params:
                layout.prop(self, name)

    Def.__name__ = 'ZensimNode_' + n_name

    def make_prop(type, defl):
        if type == 'string':
            if defl == '':
                return bpy.props.StringProperty()
            else:
                return bpy.props.StringProperty(default=defl)
        elif type == 'float':
            if defl == '':
                return bpy.props.FloatProperty()
            else:
                defl_split = list(map(float, defl.split(maxsplit=2)))
                if len(defl_split) == 1:
                    defl, = defl_split
                    return bpy.props.FloatProperty(default=defl)
                elif len(defl_split) == 2:
                    defl, minval = defl_split
                    return bpy.props.FloatProperty(default=defl, min=minval)
                elif len(defl_split) == 3:
                    defl, minval, maxval = defl_split
                    return bpy.props.FloatProperty(
                            default=defl, min=minval, max=maxval)
        elif type == 'int':
            if defl == '':
                return bpy.props.IntProperty()
            else:
                defl_split = defl.split(maxsplit=2)
                if len(defl_split) == 1:
                    return bpy.props.IntProperty(default=defl)
                elif len(defl_split) == 2:
                    defl, minval = defl_split
                    return bpy.props.IntProperty(default=defl, min=minval)
                elif len(defl_split) == 3:
                    defl, minval, maxval = defl_split
                    return bpy.props.IntProperty(
                            default=defl, min=minval, max=maxval)
        else:
            raise RuntimeError('unknown param type: ' + type)

    Def.__annotations__ = {}
    for name, type, defl in n_params:
        Def.__annotations__[name] = make_prop(type, defl)

    user_classes.append(Def)


add_zensim_node_class(
        'ReadObjMesh',
        [
            ('path', 'string', 'monkey.obj'),
            ('dx', 'float', '3.14'),
        ],
        [],
        ['mesh'],
        )


def register():
    for cls in core_classes:
        register_class(cls)
    register_nodes()


def unregister():
    unregister_nodes()
    for cls in reversed(core_classes):
        unregister_class(cls)


def register_nodes():
    for cls in user_classes:
        register_class(cls)
    nodeitems_utils.register_node_categories('ZENSIM_NODES', node_categories)


def unregister_nodes():
    try:
        nodeitems_utils.unregister_node_categories('ZENSIM_NODES')
    except KeyError:
        pass
    for cls in reversed(user_classes):
        unregister_class(cls)
