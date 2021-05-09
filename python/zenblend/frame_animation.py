import bpy

import os
import zenapi

from .blender_io import renew_mesh, renew_object, renew_volume_object
from .obj_mesh_io import readobj
from zenapi.zpmio import readzpm


curr_objects = set()


def loadFileIntoBlender(name, ext, path, frameid):
    frame_name = '{}:{:06d}'.format(name, frameid)

    if ext == '.obj':
        verts, faces = readobj(path, simple=True)
        if faces is None:
            faces = []
        else:
            faces = faces.tolist()
        verts = verts.tolist()
        mesh = renew_mesh(frame_name, pos=verts, faces=faces)
        obj = renew_object(name, mesh)

    elif ext == '.zpm':
        attrs, conns = readzpm(path)
        pos = attrs['pos'].tolist()
        faces = conns[2]
        mesh = renew_mesh(frame_name, pos=pos, faces=faces)
        obj = renew_object(name, mesh)

    elif ext == '.vdb':
        obj = renew_volume_object(name, path)

    else:
        raise RuntimeError(f'bad file extension name: {ext}')

    curr_objects.add(obj.name)


@bpy.app.handlers.persistent
def frameUpdateCallback(*unused_args):
    frameid = bpy.context.scene.frame_current

    prev_objects = frozenset(curr_objects)

    curr_objects.clear()
    for name, ext, path in zenapi.getFrameFiles(frameid):
        loadFileIntoBlender(name, ext, path, frameid)

    for name in prev_objects:
        if name not in curr_objects:
            obj = bpy.data.objects[name]
            bpy.data.objects.remove(obj)


def register():
    if frameUpdateCallback not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frameUpdateCallback)


def unregister():
    if frameUpdateCallback in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frameUpdateCallback)
