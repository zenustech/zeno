from .py import *
from .py import getNodeByName, isPyNodeName
from .api import requireObject, hasObject, newExecutionContext


portals = {}
portalIns = {}


@defNodeClass
class PortalIn(INode):
    z_inputs = ['port']
    z_params = [('string', 'name', 'RenameMe!')]
    z_categories = 'graph'

    def init(self):
        ident = self.get_node_name()
        name = self.get_param('name')
        #if name in portalIns:
        #    raise RuntimeError(f'duplicate portal name: `{name}`')
        portalIns[name] = ident

    def apply(self):
        name = self.get_param('name')
        ref = self.get_input_ref('port')
        portals[name] = ref


@defNodeClass
class PortalOut(INode):
    z_outputs = ['port']
    z_params = [('string', 'name', 'MyName')]
    z_categories = 'graph'

    def apply(self):
        name = self.get_param('name')
        if name not in portalIns:
            raise RuntimeError(f'no PortalIn for name: `{name}`')
        depnode = portalIns[name]
        requireObject(depnode + '::DST')
        ref = portals[name]
        self.set_output_ref('port', ref)


@defNodeClass
class Route(INode):
    z_inputs = ['input']
    z_outputs = ['output']
    z_categories = 'graph'

    def apply(self):
        obj = self.get_input_ref('input')
        self.set_output_ref('output', obj)


@defNodeClass
class RunOnce(INode):
    z_outputs = ['cond']
    z_categories = 'substep'

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
    z_categories = 'utility'

    def apply(self):
        import time
        secs = self.get_param('secs')
        time.sleep(secs)


@defNodeClass
class LogicConst(INode):
    z_params = [('int', 'value', '0')]
    z_outputs = ['value']
    z_categories = 'logical'

    def apply(self):
        value = self.get_param('value')
        value = BooleanObject(value)
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


@defNodeClass
class PrintMessage(INode):
    z_params = [('string', 'msg', 'Your message')]
    z_categories = 'utility'

    def apply(self):
        msg = self.get_param('msg')
        print(msg)


@defNodeClass
class IfCondition(INode):
    z_inputs = ['cond', 'true', 'false']
    z_outputs = ['out']
    z_categories = 'control'

    def apply(self):
        cond = self.get_input('cond')
        if cond:
            ret = self.get_input_ref('true')
        else:
            ret = self.get_input_ref('false')
        self.set_output_ref('out', ret)


@defNodeClass
class RepeatTimes(INode):
    z_inputs = ['stm', 'times']
    z_outputs = ['lastStm']
    z_categories = 'control'

    def apply(self):
        times = self.get_input('times')
        for i in range(times):
            with newExecutionContext():
                stm = self.get_input_ref('stm')
                self.set_output_ref('lastStm', stm)


@defNodeClass
class RepeatUntil(INode):
    z_inputs = ['stm', 'cond']
    z_outputs = ['lastStm']
    z_categories = 'control'

    def apply(self):
        while True:
            with newExecutionContext():
                if self.has_input('stm'):
                    stm = self.get_input_ref('stm')
                    self.set_output_ref('lastStm', stm)
                cond = self.get_input('cond')
                if cond:
                    break
                else:
                    print('cond not satisfied, repeat')


@defNodeClass
class CreateMutable(INode):
    z_outputs = ['mutable', 'value']
    z_categories = 'mutable'

    def apply(self):
        self.set_output('mutable', self.get_node_name())


@defNodeClass
class UpdateMutable(INode):
    z_inputs = ['mutable', 'value', 'initValue']
    z_outputs = ['value']
    z_categories = 'mutable'

    def apply(self):
        mutable = self.get_input('mutable')
        assert isPyNodeName(mutable), mutable
        mutable = getNodeByName(mutable)
        assert isinstance(mutable, CreateMutable), mutable

        hasVal = hasObject(mutable.get_output_ref('value'))
        if hasVal or not self.has_input('initValue'):
            value = self.get_input_ref('value')
        else:
            value = self.get_input_ref('initValue')
        mutable.set_output_ref('value', value)
        self.set_output_ref('value', value)


@defNodeClass
class CachedOnce(INode):
    z_inputs = ['value']
    z_outputs = ['value']
    z_categories = 'mutable'

    def apply(self):
        if not hasObject(self.get_output_ref('value')):
            value = self.get_input_ref('value')
            self.set_output_ref('value', value)


@defNodeClass
class IsNotNull(INode):
    z_inputs = ['value']
    z_outputs = ['value', 'cond']
    z_categories = 'mutable'

    def apply(self):
        ref = self.get_input_ref('value')
        cond = hasObject(ref)
        if cond:
            self.set_output_ref('value', ref)
        self.set_output('cond', BooleanObject(cond))


@defNodeClass
class MakeNull(INode):
    z_outputs = ['null']
    z_categories = 'mutable'

    def apply(self):
        pass


__all__ = []
