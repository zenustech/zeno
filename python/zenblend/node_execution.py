import bpy
from bpy.utils import register_class, unregister_class
from .compile_graph import node_graph_to_script
from .launch_script import execute_script


class ZensimExecuteOperator(bpy.types.Operator):
    '''Execute Zensim node graph'''
    bl_idname = 'node.zensim_execute'
    bl_label = 'Execute'

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR' and space.tree_type == 'ZensimTreeType'

    def execute(self, context):
        space = context.space_data
        node_tree = space.node_tree
        node_active = context.active_node
        node_selected = context.selected_nodes

        links = {}
        for link in node_tree.links:
            dst_node, dst_sock = link.to_node.name, link.to_socket.name
            src_node, src_sock = link.from_node.name, link.from_socket.name
            links[(dst_node, dst_sock)] = (src_node, src_sock)

        nodes = {}
        for node in node_tree.nodes:
            n_param_names = getattr(node, 'n_param_names', set())
            node_params = {}
            for name in n_param_names:
                if name in node:
                    value = node[name]
                else:
                    value = getattr(node, name)
                if hasattr(value, 'foreach_get'):
                    value = tuple(value)
                node_params[name] = value
            node_type = node.bl_label
            nodes[node.name] = node_type, node_params

        if 'ExecutionOutput' in node_tree.nodes:
            output_node = node_tree.nodes['ExecutionOutput']
            nframes = output_node.nframes
            wanted = {'ExecutionOutput'}
        else:
            self.report({'ERROR'}, 'Please add an ExecutionOutput node!')
            return {'CANCELED'}
        '''
        elif node_selected:
            wanted = {node.name for node in node_selected}
        else:
            wanted = {node_active.name}
        '''

        source = node_graph_to_script(links, nodes, wanted)
        execute_script(source, nframes)

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
