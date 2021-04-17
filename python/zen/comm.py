from .py import *


g = type('G', (object,), {})()
g.frame_time = 1 / 24
g.frame_time_elapsed = 0.0
g.time_step_integrated = False


@defNodeClass
class SetFrameTime(INode):
    z_params = [('float', 'time', '0.0')]
    z_categories = 'keywords'

    def apply(self):
        time = self.get_param('time')
        g.frame_time = time


@defNodeClass
class GetFrameTime(INode):
    z_outputs = ['time']
    z_categories = 'keywords'

    def apply(self):
        time = g.frame_time
        self.set_output('time', time)


@defNodeClass
class GetFrameTimeElapsed(INode):
    z_outputs = ['time']
    z_categories = 'keywords'

    def apply(self):
        time = g.frame_time_elapsed
        self.set_output('time', time)


@defNodeClass
class IntegrateFrameTime(INode):
    z_params = [('float', 'desired_dt', '0.0')]
    z_outputs = ['dt']
    z_categories = 'keywords'

    def apply(self):
        if self.has_input('desired_dt'):
            dt = self.get_input('desired_dt')
        else:
            dt = 1 / 24

        dt = min(g.frame_time - g.frame_time_elapsed, dt)

        if not g.time_step_integrated:
            g.frame_time_elapsed += dt
            g.time_step_integrated = True

        self.set_output('dt', dt)


@defNodeClass
class NumericFloat(INode):
    z_params = [('float', 'value', '0.0')]
    z_outputs = ['output']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('output', value)


@defNodeClass
class NumericInt(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['output']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('output', value)


portals = {}


@defNodeClass
class PortalIn(INode):
    z_inputs = ['port']
    z_categories = 'misc'
    z_params = [('string', 'name', 'RenameMe!')]
    def apply(self):
        id = self.get_param('name')
        obj = self.get_input('port')
        portals[id] = obj


@defNodeClass
class PortalOut(INode):
    z_categories = 'misc'
    z_outputs = ['port']
    z_params = [('string', 'name', 'MyName')]
    def apply(self):
        id = self.get_param('name')
        obj = portals[id]
        self.set_output('port', obj)


@defNodeClass
class Route(INode):
    z_inputs = ['input']
    z_outputs = ['output']
    z_categories = 'misc'

    def apply(self):
        obj = self.get_input('input')
        self.set_output('output', obj)


@defNodeClass
class RunOnce(INode):
    z_outputs = ['cond']
    z_categories = 'misc'

    def __init__(self):
        super().__init__()

        self.initialized = False

    def apply(self):
        cond = not self.initialized
        self.initialized = True
        self.set_output('cond', BooleanObject(cond))


@defNodeClass
class SleepFor(INode):
    z_params = [('float', 'secs', '1 0')]
    z_categories = 'misc'

    def apply(self):
        import time
        secs = self.get_param('secs')
        time.sleep(secs)



__all__ = []
