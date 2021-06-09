from .obj_mesh_io import readobj

import bpy


def to_numpy(b, key, dim=None):
    dim = dim or len(getattr(b[0], key))
    dim = len(getattr(b[0], key))
    seq = [0] * (len(b) * dim)
    b.foreach_get(key, seq)
    return np.array(seq).reshape(len(b), dim)


def from_numpy(b, key, a, dim=None):
    a = np_array(a)
    if dim is None:
        dim = len(getattr(b[0], key)) if len(b) else a.shape[1]
    assert len(a.shape) == 2
    assert a.shape[1] == dim
    if len(b) < a.shape[0]:
        b.add(a.shape[0] - len(b))
    seq = a.reshape(a.shape[0] * dim).tolist()
    seq = seq + [0] * (len(b) * dim - len(seq))
    b.foreach_set(key, seq)


def new_mesh(name, pos=[], edges=[], faces=[], uv=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(pos, edges, faces)
    if uv is not None:
        mesh.uv_layers.new()
        from_numpy(mesh.uv_layers.active.data, 'uv', uv)
    return mesh


def renew_mesh(name, pos=[], edges=[], faces=[], uv=None):
    if name in bpy.data.meshes:
        mesh = bpy.data.meshes[name]
        bpy.data.meshes.remove(mesh)
    mesh = new_mesh(name, pos, edges, faces, uv)
    return mesh


def new_object(name, mesh):
    obj = bpy.data.objects.new(name, mesh)
    col = bpy.context.collection
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def renew_object(name, mesh):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        obj.data = mesh
    else:
        obj = new_object(name, mesh)
    return obj


def renew_volume_object(name, path):
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        bpy.data.objects.remove(obj)
    bpy.ops.object.volume_import(filepath=path)
    obj = bpy.context.object
    obj.name = name
    return obj


def mesh_update(mesh, pos=None, edges=None, faces=None):
    if pos is not None:
        from_numpy(mesh.vertices, 'co', pos)
    if edges is not None:
        from_numpy(mesh.edges, 'vertices', edges)
    if faces is not None:
        from_numpy(mesh.polygons, 'vertices', faces)
    mesh.update()


__all__ = []
