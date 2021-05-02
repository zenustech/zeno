import zenlibs

core = zenlibs.get_library('libzenvis')


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


def uploadStatus():
    core.set_window_size(*status['resolution'])
    core.look_perspective(*status['perspective'])
    core.set_curr_playing(status['playing'])
    if status['next_frameid'] != -1:
        core.set_curr_frameid(status['next_frameid'])


def _recieveStatus():
    frameid = core.get_curr_frameid()
    solver_frameid = core.get_solver_frameid()
    solver_interval = core.get_solver_interval()
    render_fps = core.get_render_fps()

    status.update({
        'frameid': frameid,
        'solver_frameid': solver_frameid,
        'solver_interval': solver_interval,
        'render_fps': render_fps,
    })


def initializeGL():
    core.initialize()


def paintGL():
    core.new_frame()
    _recieveStatus()
