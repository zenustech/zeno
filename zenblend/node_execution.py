import bpy
from bpy.utils import register_class, unregister_class

from zeno import launch
from zenutils import go



def dumpBlenderGraph(node_tree):
    links = {}
    for link in node_tree.links:
        dst_node, dst_sock = link.to_node.name, link.to_socket.name
        src_node, src_sock = link.from_node.name, link.from_socket.name
        links[dst_node, dst_sock] = src_node, src_sock

    nodes = {}
    for node in node_tree.nodes:
        n_param_names = getattr(node, 'n_param_names', [])
        n_input_names = getattr(node, 'n_input_names', [])

        node_params = {}
        for name in n_param_names:
            if name in node:
                value = node[name]
            else:
                value = getattr(node, name)
            if hasattr(value, 'foreach_get'):  # is bpy_prop_array (vector)
                value = tuple(value)
            node_params[name] = value

        node_inputs = {}
        for name in n_input_names:
            node_inputs[name] = links.get((node.name, name), None)

        node_type = node.bl_label
        node_uipos = ''
        nodes[node.name] = {
            'name': node_type,
            'inputs': node_inputs,
            'params': node_params,
            'uipos': node_uipos,
            'options': ['OUT'] if node_type == 'ExecutionOutput' else [],
        }

    return nodes


class ZensimExecuteOperator(bpy.types.Operator):
    '''Execute Zensim node graph'''
    bl_idname = 'node.zensim_execute'
    bl_label = 'Execute'

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR' and space.tree_type == 'ZensimTreeType'

    def execute(self, context):
        scene = context.scene
        space = context.space_data
        node_tree = space.node_tree
        node_active = context.active_node
        node_selected = context.selected_nodes

        if 'ExecutionOutput' in node_tree.nodes:
            output_node = node_tree.nodes['ExecutionOutput']
            nframes = output_node.nframes
        else:
            self.report({'ERROR'}, 'Please add an ExecutionOutput node!')
            return {'CANCELED'}

        graph = dumpBlenderGraph(node_tree)
        scene = {'main': graph}
        go(launch.launchScene, scene, nframes)

        return {'FINISHED'}


classes = [
    ZensimExecuteOperator,
]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
