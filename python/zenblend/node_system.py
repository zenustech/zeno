import bpy
from bpy.utils import register_class, unregister_class
from bpy.types import NodeTree, Node, NodeSocket

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem



class ZensimTree(NodeTree):
    '''Zensim node system for physics simulation'''
    bl_idname = 'ZensimTreeType'
    bl_label = 'Zensim Node Tree'
    bl_icon = 'PHYSICS'


class ZensimTreeNode(Node):
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'ZensimTreeType'


class ZensimSocket(NodeSocket):
    '''Zensim socket'''
    bl_idname = 'ZensimSocketType'
    bl_label = 'Socket'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (1.0, 0.4, 0.216, 1.0)


class ZensimNodeCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR' and space.tree_type == 'ZensimTreeType'


def to_identifier_upper(s):
    return s.upper().replace(' ', '_')


core_classes = [
    ZensimTree,
    ZensimSocket,
]

node_categories = []
user_classes = []
user_categories = {}


def add_zensim_node_class(line):
    line = line.strip()
    if ':' not in line:
        return
    n_name, rest = line.split(':', maxsplit=1)
    assert rest.startswith('(') and rest.endswith(')'), (n_name, rest)
    inputs, outputs, params, category = rest.strip('()').split(')(')

    n_inputs = [name for name in inputs.split(',') if name]
    n_outputs = [name for name in outputs.split(',') if name]
    n_params = []
    for param in params.split(','):
        if not param:
            continue
        type, name, defl = param.split(':')
        n_params.append((type, name, defl))

    #print('[ZenBlend] registering:', n_name, n_inputs, n_outputs, n_params)
    do_add_zensim_node_class(n_name, n_inputs, n_outputs, n_params, category)


def do_add_zensim_node_class(n_name, n_inputs, n_outputs, n_params, category):
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
            for type, name, defl in n_params:
                layout.prop(self, name)

    Def.__name__ = 'ZensimNode_' + n_name

    Def.__annotations__ = {}
    for type, name, defl in n_params:
        Def.__annotations__[name] = make_zensim_param_property(type, defl)

    user_classes.append(Def)
    user_categories.setdefault(category, []).append(Def.bl_idname)


def make_zensim_param_property(type, defl):
    if type == 'string':
        if not defl:
            return bpy.props.StringProperty()
        else:
            return bpy.props.StringProperty(default=defl)

    elif type == 'float':
        if not defl:
            return bpy.props.FloatProperty()
        else:
            defl_split = [float(x) for x in defl.split(maxsplit=2)]
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
            else:
                assert False, defl_split

    elif type == 'int':
        if not defl:
            return bpy.props.IntProperty()
        else:
            defl_split = [int(x) for x in defl.split(maxsplit=2)]
            if len(defl_split) == 1:
                defl, = defl_split
                return bpy.props.IntProperty(default=defl)
            elif len(defl_split) == 2:
                defl, minval = defl_split
                return bpy.props.IntProperty(default=defl, min=minval)
            elif len(defl_split) == 3:
                defl, minval, maxval = defl_split
                return bpy.props.IntProperty(
                        default=defl, min=minval, max=maxval)
            else:
                assert False, defl_split

    elif type == 'float3':
        if not defl:
            return bpy.props.FloatVectorProperty()
        else:
            defl_split = [float(x) for x in defl.split(maxsplit=2)]
            assert len(defl_split) == 3, defl_split
            x, y, z = defl_split
            return bpy.props.FloatVectorProperty(default=(x, y, z))

    elif type == 'int3':
        if not defl:
            return bpy.props.IntVectorProperty()
        else:
            defl_split = [int(x) for x in defl.split(maxsplit=2)]
            assert len(defl_split) == 3, defl_split
            x, y, z = defl_split
            return bpy.props.IntVectorProperty(default=(x, y, z))

    else:
        assert False, 'unknown param type: ' + type


def generate_node_categories_from_user_categories():
    node_categories.clear()
    for cate_name, node_names in user_categories.items():
        cate_id_name = to_identifier_upper(cate_name)
        items = [NodeItem(node_name) for node_name in node_names]
        category = ZensimNodeCategory(cate_id_name, cate_name, items=items)
        node_categories.append(category)


class ZensimNode_ExecutionOutput(ZensimTreeNode):
    '''Zensim graph execution output'''

    category = 'blender'
    bl_idname = 'ZensimNodeType_ExecutionOutput'
    bl_label = 'ExecutionOutput'
    bl_icon = 'PHYSICS'

    def init(self, context):
        self.inputs.new('ZensimSocketType', 'SRC')

    def draw_buttons(self, context, layout):
        layout.operator('node.zensim_execute')


nonproc_class = [
    ZensimNode_ExecutionOutput,
]


def load_user_nodes_from_descriptors(descriptors):
    unregister_user_nodes()

    user_classes.clear()
    user_categories.clear()
    for line in descriptors.splitlines():
        add_zensim_node_class(line)

    for cls in nonproc_class:
        user_classes.append(cls)
        user_categories.setdefault(cls.category, []).append(cls.bl_idname)

    generate_node_categories_from_user_categories()

    register_user_nodes()


user_nodes_registered = False


def register_user_nodes():
    global user_nodes_registered

    for cls in user_classes:
        register_class(cls)
    nodeitems_utils.register_node_categories('ZENSIM_NODES', node_categories)

    user_nodes_registered = True


def unregister_user_nodes():
    global user_nodes_registered
    if not user_nodes_registered:
        return

    try:
        nodeitems_utils.unregister_node_categories('ZENSIM_NODES')
    except KeyError:
        pass
    node_categories.clear()
    for cls in reversed(user_classes):
        unregister_class(cls)

    user_nodes_registered = False


def register():
    for cls in core_classes:
        register_class(cls)

    load_user_nodes_from_descriptors('''
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
''')


def unregister():
    unregister_user_nodes()
    for cls in reversed(core_classes):
        unregister_class(cls)
