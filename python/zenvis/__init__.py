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


def _frameUpdate():
    max_frameid = zenapi.getFrameCount(250)
    frameid = core.play_frameid(max_frameid)
    for name, ext, path in zenapi.getFrameFiles(frameid):
        core.load_file(name, ext, path, frameid)


def initializeGL():
    core.initialize()


def paintGL():
    _uploadStatus()
    _frameUpdate()
    core.new_frame()
    _recieveStatus()
