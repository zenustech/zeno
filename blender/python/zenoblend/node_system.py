import bpy

from .cpp_dll import core

for i, name in enumerate(core.get_updated_names()):
    zs_mesh = core.get_updated_mesh(i)
    nverts = core.get_mesh_verts(zs_mesh, 0)
    if nverts:
        mesh.vertices.add(nverts)
        core.get_mesh_verts(mesh.vertices[0].as_pointer())