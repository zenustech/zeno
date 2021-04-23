from .py import *
import numpy as np

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


@defNodeClass
class NumericFloat(INode):
    z_params = [('float', 'value', '0.0')]
    z_outputs = ['value']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('value', value)


@defNodeClass
class NumericVec2(INode):
    z_params = [('float', 'x', '0.0'), ('float', 'y', '0.0')]
    z_outputs =['vec2']
    z_categories = 'numeric'

    def apply(self):
        vec2 = np.array([self.get_param('x'), self.get_param('y')])
        self.set_output('vec2', vec2)


@defNodeClass
class NumericVec3(INode):
    z_params = [('float', 'x', '0.0'), ('float', 'y', '0.0'), ('float', 'z', '0.0')]
    z_outputs =['vec3']
    z_categories = 'numeric'

    def apply(self):
        vec3 = np.array([self.get_param('x'), self.get_param('y'), self.get_param('z')])
        self.set_output('vec3', vec3)


@defNodeClass
class NumericVec4(INode):
    z_params = [('float', 'x', '0.0'), ('float', 'y', '0.0'), ('float', 'z', '0.0'), ('float', 'w', '0.0')]
    z_outputs =['vec4']
    z_categories = 'numeric'

    def apply(self):
        vec4 = np.array([self.get_param('x'), self.get_param('y'), self.get_param('z'), self.get_param('w'),])
        self.set_output('vec4', vec4)


@defNodeClass
class NumericInt(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['value']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('value', value)


@defNodeClass
class PrintNumeric(INode):
    z_inputs = ['value']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_input('value')
        print('PrintNumeric:', value)



__all__ = []
