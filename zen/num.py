from .py import *

import numpy as np


@defNodeClass
class NumericInt(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['value']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('value', value)


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
        vec2 = [self.get_param('x'), self.get_param('y')]
        self.set_output('vec2', vec2)


@defNodeClass
class NumericVec3(INode):
    z_params = [('float', 'x', '0.0'), ('float', 'y', '0.0'), ('float', 'z', '0.0')]
    z_outputs =['vec3']
    z_categories = 'numeric'

    def apply(self):
        vec3 = [self.get_param('x'), self.get_param('y'), self.get_param('z')]
        self.set_output('vec3', vec3)


@defNodeClass
class NumericVec4(INode):
    z_params = [('float', 'x', '0.0'), ('float', 'y', '0.0'), ('float', 'z', '0.0'), ('float', 'w', '0.0')]
    z_outputs =['vec4']
    z_categories = 'numeric'

    def apply(self):
        vec4 = [self.get_param('x'), self.get_param('y'), self.get_param('z'), self.get_param('w')]
        self.set_output('vec4', vec4)


@defNodeClass
class NumericInt(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['value']
    z_categories = 'numeric'

    def apply(self):
        value = self.get_param('value')
        self.set_output('value', value)


BIN_OP_TABLE = dict(
        add=np.add,
        sub=np.subtract,
        mul=np.multiply,
        div=np.divide,
        pow=np.power,
        atan2=np.arctan2,
        min=np.minimum,
        max=np.maximum,
        )
UN_OP_TABLE = dict(
        sin=np.sin,
        cos=np.cos,
        tan=np.tan,
        asin=np.arcsin,
        acos=np.arccos,
        atan=np.arctan,
        sqrt=np.sqrt,
        exp=np.exp,
        log=np.log,
        max=np.max,
        min=np.min,
        )

def do_binary_operator(op, lhs, rhs):
    lhs, rhs = np.array(lhs), np.array(rhs)
    if op not in BIN_OP_TABLE:
        raise RuntimeError(f'bad binary operator type: {op}')
    ret = BIN_OP_TABLE[op](lhs, rhs)
    return ret.tolist()

def do_unary_operator(op, lhs):
    lhs = np.array(lhs)
    if op not in UN_OP_TABLE:
        raise RuntimeError(f'bad unary operator type: {op}')
    ret = UN_OP_TABLE[op](lhs)
    return ret.tolist()


@defNodeClass
class NumericOperator(INode):
    z_params = [('string', 'op_type', '')]
    z_inputs = ['lhs', 'rhs']
    z_outputs = ['ret']
    z_categories = 'numeric'

    def apply(self):
        op = self.get_param('op_type')
        lhs = self.get_input('lhs')
        if self.has_input('rhs'):
            rhs = self.get_input('rhs')
            ret = do_binary_operator(op, lhs, rhs)
        else:
            ret = do_unary_operator(op, lhs)
        self.set_output('ret', ret)


@defNodeClass
class PrintNumeric(INode):
    z_inputs = ['value']
    z_params = [('string', 'hint', 'PrintNumeric')]
    z_categories = 'numeric'

    def apply(self):
        value = self.get_input('value')
        hint = self.get_param('hint')
        print('[{}] {!r}'.format(hint, value))


__all__ = []
