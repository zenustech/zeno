import zen


@zen.defNodeClass
class MyNode(zen.INode):
    z_name = 'MyNode'
    z_params = [('dx', 'float', '0 0')]
    z_inputs = ['lhs', 'rhs']
    z_outputs = ['out']

    def apply(self):
        dx = self.get_param('dx')
        lhs = self.get_input('lhs')
        rhs = self.get_input('rhs')
        out = lhs * dx + rhs
        self.set_output('out', out)
