from .py import *


@defNodeClass
class MakeString(INode):
    z_params = [('string', 'value', '')]
    z_outputs = ['value']
    z_categories = 'imexport'

    def apply(self):
        value = self.get_param('value')
        self.set_output('value', value)


@defNodeClass
class AskExport(INode):
    z_inputs = ['path']
    z_categories = 'imexport'

    def apply(self):
        path = self.get_input('path')
        with open('/tmp/zenexport', 'a') as f:
            print(path, file=f)


__all__ = []
