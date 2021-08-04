import bpy
from bpy.utils import register_class, unregister_class
from bpy.types import NodeTree, Node, NodeSocket

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

from zeno import launch



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


def do_add_zensim_node_class(n_name, n_inputs, n_outputs, n_params, category):
    class Def(ZensimTreeNode):
        __doc__ = 'Zensim node: ' + n_name
        bl_idname = 'ZensimNodeType_' + n_name
        bl_label = n_name
        bl_icon = 'BLENDER' if category == 'blender' else 'PHYSICS'
        n_param_names = [name for type, name, defl in n_params]
        n_input_names = [name for name in n_inputs]

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
    bl_icon = 'BLENDER'
    n_param_names = []
    n_input_names = ['SRC']

    nframes: bpy.props.IntProperty(min=1, default=1,
            description='Number of frames to execute')

    def init(self, context):
        self.inputs.new('ZensimSocketType', 'SRC')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'nframes')
        layout.operator('node.zensim_execute')


nonproc_class = [
    ZensimNode_ExecutionOutput,
]


def load_user_nodes_from_descriptors(descs):
    unregister_user_nodes()

    user_classes.clear()
    user_categories.clear()
    for n_name, desc in descs.items():
        do_add_zensim_node_class(n_name, desc['inputs'], desc['outputs'],
                desc['params'], desc['categories'][0])

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
    descs = launch.getDescriptors()
    load_user_nodes_from_descriptors(descs)
    for cls in core_classes:
        register_class(cls)


def unregister():
    unregister_user_nodes()
    for cls in reversed(core_classes):
        unregister_class(cls)
