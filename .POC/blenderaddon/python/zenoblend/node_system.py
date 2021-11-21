import bpy

from .cpp_dll import core


our_creations = set()


def create_bl_mesh(name):
    if name not in bpy.data.meshes:
        mesh = bpy.data.meshes.new(name)
    else:
        mesh = bpy.data.meshes[name]
        mesh.clear_geometry()
        our_creations.add(name)
    return mesh


def zs_mesh_to_bl(zs_mesh, bl_mesh):
    nverts = core.get_mesh_verts(zs_mesh, 0)
    if nverts:
        bl_mesh.vertices.add(nverts)
        core.get_mesh_verts(zs_mesh, bl_mesh.vertices[0].as_pointer())


for i in range(core.get_updates_count()):
    bl_name = core.get_update_bl_name(zs_mesh, 0)
    bl_mesh = create_bl_mesh(bl_name)
    zs_mesh = core.get_update_zs_mesh(i)
    zs_mesh_to_bl(zs_mesh, bl_mesh)