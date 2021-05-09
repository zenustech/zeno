from .py import *


portals = {}


@defNodeClass
class PortalIn(INode):
    z_inputs = ['port']
    z_categories = 'misc'
    z_params = [('string', 'name', 'RenameMe!')]
    def apply(self):
        id = self.get_param('name')
        ref = self.get_input_ref('port')
        portals[id] = ref


@defNodeClass
class PortalOut(INode):
    z_categories = 'misc'
    z_outputs = ['port']
    z_params = [('string', 'name', 'MyName')]
    def apply(self):
        id = self.get_param('name')
        ref = portals[id]
        self.set_output_ref('port', ref)


@defNodeClass
class Route(INode):
    z_inputs = ['input']
    z_outputs = ['output']
    z_categories = 'misc'

    def apply(self):
        obj = self.get_input_ref('input')
        self.set_output_ref('output', obj)


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
