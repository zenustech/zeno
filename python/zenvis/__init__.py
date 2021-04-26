@eval('lambda x: x()')
def core():
    import os
    import sys

    lib_dir = os.path.dirname(__file__)
    lib_path = os.path.join(lib_dir, 'libzenvis.so')
    assert os.path.exists(lib_path), lib_path

    sys.path.insert(0, lib_dir)
    try:
        import libzenvis as core
    finally:
        assert sys.path.pop(0) == lib_dir

    return core


dnStat = {
    'frameid': 0,
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
}

upStat = {
    'next_frameid': -1,
    'resolution': (1, 1),
    'perspective': (),
    'playing': True,
}


def uploadStatus():
    core.set_window_size(*upStat['resolution'])
    core.look_perspective(*upStat['perspective'])
    core.set_curr_playing(upStat['playing'])
    if upStat['next_frameid'] != -1:
        core.set_curr_frameid(upStat['next_frameid'])


def _recieveStatus():
    frameid = core.get_curr_frameid()
    solver_frameid = core.get_solver_frameid()
    solver_interval = core.get_solver_interval()
    render_fps = core.get_render_fps()

    dnStat.update({
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
