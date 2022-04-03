from ...bin import pylib_zenvis as core

from ...system import fileio

from ..keyframe_editor.frame_curve_editor import lerp
from ..keyframe_editor.curve_canvas import Bezier

status = {
    'solver_frameid': 0,
    'solver_interval': 0,
    'render_fps': 0,
    'resolution': (1, 1),
    'perspective': (),
    'cache_frames': 3,
    'show_grid': True,
    'playing': True,
    'target_frame': 0,
    'lights': {},
    'prev_frame': -1,
    'camera_keyframes': {},
}

camera_keyframe = None
camera_control = None

def _uploadStatus():
    core.set_window_size(*status['resolution'])
    core.look_perspective(*status['perspective'])
    _uploadLights()

def query_value(data, x: float, sel_channel: str) -> float:
    ps = data[sel_channel]
    if len(ps) == 0:
        return 0
    if x < ps[0].pos.x or x > ps[-1].pos.x:
        return ps[-1].pos.y
    i = len(list(filter(lambda p: p.pos.x <= x, ps))) - 1
    p1 = ps[i].pos
    if p1.x == x or ps[i].cp_type == 'constant':
        return p1.y
    elif ps[i].cp_type == 'straight':
        p2 = ps[i + 1].pos
        t = (x - p1.x) / (p2.x - p1.x)
        return lerp(p1.y, p2.y, t)
    else:
        p2 = ps[i + 1].pos
        h1 = ps[i].pos + ps[i].right_handler
        h2 = ps[i+1].pos + ps[i+1].left_handler
        b = Bezier(p1, p2, h1, h2)
        return b.query(x)

def _uploadLights():
    f = get_curr_frameid()
    if status['prev_frame'] != f:
        status['prev_frame'] = f
        for index, l in status['lights'].items():
            x = query_value(l, f, 'DirX')
            y = query_value(l, f, 'DirY')
            z = query_value(l, f, 'DirZ')
            height = query_value(l, f, 'Height')
            softness = query_value(l, f, 'Softness')
            sr = query_value(l, f, 'ShadowR')
            sg = query_value(l, f, 'ShadowG')
            sb = query_value(l, f, 'ShadowB')
            cr = query_value(l, f, 'ColorR')
            cg = query_value(l, f, 'ColorG')
            cb = query_value(l, f, 'ColorB')
            intensity = query_value(l, f, 'Intensity')
            core.setLightData(
                index,
                (x, y, z),
                height,
                softness,
                (sr, sg, sb),
                (cr, cg, cb),
                intensity,                
            )

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
        core.clearReflectMask()
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
