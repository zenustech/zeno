import os

from . import launch


def getFrameFiles(frameid):
    dirpath = os.path.join(launch.iopath, '{:06d}'.format(frameid))
    if not os.path.isdir(dirpath):
        return ()
    for name in os.listdir(dirpath):
        path = os.path.join(dirpath, name)
        name, ext = os.path.splitext(os.path.basename(path))
        yield name, ext, path


def getFrameCount(max_frameid):
    for frameid in range(max_frameid):
        dirpath = os.path.join(launch.iopath, '{:06d}'.format(frameid))
        if not os.path.isdir(dirpath):
            return frameid
    return max_frameid
