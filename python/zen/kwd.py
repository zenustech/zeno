'''
Frame & substep control

(pyb to zxx: actually should rename keywords -> stepcontrol?)
'''


from .kwd import *
from .py import *
from .api import *


class G:
    pass

G = G()
G.frameid = 0
G.substepid = 0
G.frame_time = 0.03
G.frame_time_elapsed = 0.0
G.has_frame_completed = False
G.has_substep_executed = False
G.time_step_integrated = False


def substepShouldContinue():
    if G.has_substep_executed:
        if not G.time_step_integrated:
            return False
    return not G.has_frame_completed

def frameBegin():
    G.has_frame_completed = False
    G.has_substep_executed = False
    G.has_time_step_integrated = False
    G.frame_time_elapsed = 0.0

    if G.frameid == 0: addNode('EndFrame', 'endFrame')

def frameEnd():
    applyNode('endFrame')

    G.frameid += 1

def substepBegin():
    pass

def substepEnd():
    G.substepid += 1
    G.has_substep_executed = True


@defNodeClass
class RunBeforeFrame(INode):
    z_outputs = ['cond']
    z_categories = 'misc'

    def apply(self):
        cond = G.has_substep_executed
        self.set_output('cond', BooleanObject(cond))


@defNodeClass
class RunAfterFrame(INode):
    z_outputs = ['cond']
    z_categories = 'misc'

    def apply(self):
        cond = G.has_frame_completed
        self.set_output('cond', BooleanObject(cond))


@defNodeClass
class SetFrameTime(INode):
    z_inputs = ['time']
    z_categories = 'keywords'

    def apply(self):
        time = self.get_input('time')
        G.frame_time = time


@defNodeClass
class GetFrameTime(INode):
    z_outputs = ['time']
    z_categories = 'keywords'

    def apply(self):
        time = G.frame_time
        self.set_output('time', time)


@defNodeClass
class GetFrameTimeElapsed(INode):
    z_outputs = ['time']
    z_categories = 'keywords'

    def apply(self):
        time = G.frame_time_elapsed
        self.set_output('time', time)


@defNodeClass
class IntegrateFrameTime(INode):
    z_inputs = ['desired_dt']
    z_outputs = ['actual_dt']
    z_params = [('float', 'min_scale', '0.0001')]
    z_categories = 'keywords'

    def apply(self):
        dt = G.frame_time
        if self.has_input('desired_dt'):
            dt = self.get_input('desired_dt')
            min_scale = self.get_param('min_scale')
            dt = abs(dt)
            dt = max(dt, min_scale * G.frame_time)

        if G.frame_time_elapsed + dt >= G.frame_time:
            dt = G.frame_time - G.frame_time_elapsed
            G.frame_time_elapsed = G.frame_time
            G.has_frame_completed = True
        else:
            G.frame_time_elapsed += dt

        G.time_step_integrated = True
        self.set_output('actual_dt', dt)


__all__ = [
    'G',
    'substepShouldContinue',
    'frameBegin',
    'frameEnd',
    'substepBegin',
    'substepEnd',
]
