import zenlibs
import zenapi
from zenapi.zpmio import readzpm

core = zenlibs.import_library('libzenvis')


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


old_frameid = -2

def _frameUpdate():
    global old_frameid
    max_frameid = zenapi.getFrameCount(250)
    frameid = core.get_curr_frameid()
    if status['playing']:
        frameid += 1
    frameid = min(frameid, max_frameid - 1)
    core.set_curr_frameid(frameid)
    if old_frameid != frameid:
        core.clear_graphics()
        for name, ext, path in zenapi.getFrameFiles(frameid):
            core.load_file(name, ext, path, frameid)
    old_frameid = frameid


def initializeGL():
    core.initialize()


def paintGL():
    _uploadStatus()
    _frameUpdate()
    core.new_frame()
    _recieveStatus()
