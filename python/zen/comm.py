from .py import *


@defNodeClass
class Route(INode):
    z_inputs = ['input']
    z_outputs = ['output']
    z_categories = 'misc'

    def apply(self):
        obj = self.get_input('input')
        print('Route', obj)
        self.set_output('output', obj)



__all__ = []
