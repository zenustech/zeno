import bpy

import os
import zenapi

from .blender_io import renew_mesh, renew_object
from .obj_mesh_io import readobj


curr_objects = set()
prev_objects = frozenset()


def loadFileIntoBlender(name, ext, path, frameid):
    assert ext == '.obj'

    verts, faces = readobj(path, simple=True)
    if faces is None:
        faces = []
    else:
        faces = faces.tolist()
    verts = verts.tolist()

    mesh = renew_mesh('{}:{:06d}'.format(name, frameid), pos=verts, faces=faces)
    obj = renew_object(name, mesh)
    curr_objects.add(obj.name)


@bpy.app.handlers.persistent
def frameUpdateCallback(*unused_args):
    frameid = bpy.context.scene.frame_current

    global prev_objects
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
