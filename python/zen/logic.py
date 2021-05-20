from .py import *


@defNodeClass
class LogicConst(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['value']
    z_categories = 'logical'

    def apply(self):
        value = self.get_param('value')
        value = bool(value)
        self.set_output('value', value)


@defNodeClass
class LogicAnd(INode):
    z_inputs = ['lhs', 'rhs']
    z_outputs = ['res']
    z_categories = 'logical'

    def apply(self):
        lhs = self.get_param('lhs')
        rhs = self.get_param('rhs')
        res = lhs and rhs
        self.set_output('res', res)


@defNodeClass
class LogicOr(INode):
    z_inputs = ['lhs', 'rhs']
    z_outputs = ['res']
    z_categories = 'logical'

    def apply(self):
        lhs = self.get_param('lhs')
        rhs = self.get_param('rhs')
        res = lhs or rhs
        self.set_output('res', res)


@defNodeClass
class LogicNot(INode):
    z_inputs = ['val']
    z_outputs = ['res']
    z_categories = 'logical'

    def apply(self):
        val = self.get_param('val')
        res = not val
        self.set_output('res', res)


__all__ = []
