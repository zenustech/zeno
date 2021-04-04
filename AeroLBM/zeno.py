class IObject:
    pass


class INode:
    def __init__(self):
        self.params = {}
        self.inputs = {}
        self.outputs = {}
        self.initialized = False

    def initialize(self):
        pass

    def apply(self):
        raise NotImplementedError


class Session:
    def __init__(self):
        self.nodes = {}
        self.types = {}
        self.objects = {}

    def addNode(self, type, name):
        node = self.types[type]()
        self.nodes[name] = node

    def setNodeParam(self, name, key, value):
        node = self.nodes[name]
        node.params[key] = value

    def setNodeInput(self, name, key, srcname):
        node = self.nodes[name]
        value = self.objects[srcname]
        node.inputs[key] = value

    def applyNode(self, name):
        node = self.nodes[name]
        node.apply()
        for key, value in node.outputs.items():
            objname = name + '::' + key
            self.objects[objname] = value

    def initNode(self, name):
        node = self.nodes[name]
        node.initialize()
        for key, value in node.outputs.items():
            objname = name + '::' + key
            self.objects[objname] = value

    def defNodeClass(self, name):
        def decorator(type):
            self.types[name] = type
            return type

        if not isinstance(name, str):
            type = name
            name = type.__name__
            return decorator(type)
        return decorator

    def getObject(self, name):
        return self.objects[name]

session = Session()

addNode = session.addNode
initNode = session.initNode
setNodeParam = session.setNodeParam
setNodeInput = session.setNodeInput
applyNode = session.applyNode
defNodeClass = session.defNodeClass
getObject = session.getObject


'''
EXAMPLE:

@defNodeClass
class Add(INode):
    descriptor = dict(
        inputs=['lhs', 'rhs'],
        outputs=['res'],
    )

    def apply(self):
        lhs = self.inputs['lhs']
        rhs = self.inputs['rhs']
        res = lhs + rhs
        self.outputs['res'] = res
'''