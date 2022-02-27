from ...bin import pylib_zenvis as core

from ...system import fileio


status = {
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
    'resolution': (1, 1),
    'perspective': (),
    'cache_frames': 10,
    'show_grid': True,
    'playing': True,
    'target_frame': 0,
}

camera_keyframe = None
camera_control = None

def _uploadStatus():
    core.set_window_size(*status['resolution'])
    core.look_perspective(*status['perspective'])

def _recieveStatus():
    frameid = core.get_curr_frameid()
    solver_interval = core.get_solver_interval()
    render_fps = core.get_render_fps()

    status.update({
        'solver_interval': solver_interval,
        'render_fps': render_fps,
    })


old_frame_files = ()

def _frameUpdate():
    if fileio.isIOPathChanged():
        core.clear_graphics()

    frameid = get_curr_frameid()
    if status['playing']:
        frameid += 1
    frameid = set_curr_frameid(frameid)
    core.auto_gc_frame_data(status['cache_frames'])
    #print(core.get_valid_frames_list())
    core.set_show_grid(status['show_grid'])

    global old_frame_files
    frame_files = fileio.getFrameFiles(frameid)
    if old_frame_files != frame_files:
        for name, ext, path in frame_files:
            core.load_file(name, ext, path, frameid)
    old_frame_files = frame_files


def initializeGL():
    core.initialize()


def paintGL():
    _frameUpdate()
    _uploadStatus()
    core.new_frame()
    _recieveStatus()

def recordGL(path):
    core.set_window_size(*status['resolution'])
    core.look_perspective(*status['perspective'])
    core.new_frame_offline(path)

def get_curr_frameid():
    return core.get_curr_frameid()

def set_curr_frameid(frameid):
    start, count = fileio.getFrameRange()
    if frameid < start:
        frameid = start
    if frameid >= start + count:
        frameid = start + count - 1
    cur_frameid = core.get_curr_frameid()
    core.set_curr_frameid(frameid)
    if cur_frameid != frameid and camera_keyframe != None and camera_control != None:
        r = camera_keyframe.query_frame(frameid)
        if r:
            # print(r)
            camera_control.set_keyframe(r)
            camera_control.update_perspective()

    return frameid
