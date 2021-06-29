import zenapi
from zenapi.zpmio import readzpm

from . import pyzenvis as core


status = {
    'frameid': 0,
    'next_frameid': -1,
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
    'resolution': (1, 1),
    'perspective': (),
    'playing': True,
}


def _uploadStatus():
    core.set_window_size(*status['resolution'])
    core.look_perspective(*status['perspective'])
    core.set_curr_playing(status['playing'])
    if status['next_frameid'] != -1:
        core.set_curr_frameid(status['next_frameid'])


def _recieveStatus():
    frameid = core.get_curr_frameid()
    solver_interval = core.get_solver_interval()
    render_fps = core.get_render_fps()

    status.update({
        'frameid': frameid,
        'solver_interval': solver_interval,
        'render_fps': render_fps,
    })


old_frame_files = ()

def _frameUpdate():
    max_frameid = zenapi.getFrameCount()
    frameid = core.get_curr_frameid()
    if status['playing']:
        frameid += 1
    frameid = min(frameid, max_frameid - 1)
    core.set_curr_frameid(frameid)

    global old_frame_files
    frame_files = zenapi.getFrameFiles(frameid)
    if old_frame_files != frame_files:
        core.clear_graphics()
        for name, ext, path in frame_files:
            core.load_file(name, ext, path, frameid)
    old_frame_files = frame_files


def initializeGL():
    core.initialize()


def paintGL():
    _uploadStatus()
    _frameUpdate()
    core.new_frame()
    _recieveStatus()
