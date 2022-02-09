import os

from . import launch

g_lastiopath = None

def isIOPathChanged():
    global g_lastiopath
    res = g_lastiopath != launch.g_iopath
    g_lastiopath = launch.g_iopath
    return res


def getFrameFiles(frameid):
    if launch.g_iopath is None:
        return ()
    dirpath = os.path.join(launch.g_iopath, '{:06d}'.format(frameid))
    if not os.path.exists(os.path.join(dirpath, 'done.lock')):
        return ()
    res = []
    for name in os.listdir(dirpath):
        path = os.path.join(dirpath, name)
        name, ext = os.path.splitext(os.path.basename(path))
        if os.path.exists(path):
            res.append((name, ext, path))
    return tuple(res)


def getFrameCount(max_frameid=None):
    if launch.g_iopath is None:
        return 0
    frameid = 0
    while max_frameid is None or frameid < max_frameid:
        dirpath = os.path.join(launch.g_iopath, '{:06d}'.format(frameid))
        if not os.path.exists(os.path.join(dirpath, 'done.lock')):
            return frameid
        frameid += 1
    return max_frameid

# return value: tuple: (frame_start, frame_count)
def getFrameRange():
    if launch.g_iopath is None:
        return (0, 0)
    ns = os.listdir(launch.g_iopath)
    ns = [int(n) for n in ns if n.isdigit()]
    if len(ns) == 0:
        return (0, 0)
    ns.sort()
    frame_start = ns[0]
    frame_count = 0

    for n in ns:
        dirpath = os.path.join(launch.g_iopath, '{:06d}'.format(n))
        if os.path.exists(os.path.join(dirpath, 'done.lock')):
            frame_count += 1
        else:
            break
    return (frame_start, frame_count)
