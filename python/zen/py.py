'''
Python APIs
'''


import abc
from collections import namedtuple


class IObject:
    pass


class BooleanObject(int, IObject):
    pass


class ParamDescriptor(namedtuple('ParamDescriptor',
    ['type', 'name', 'defl'])):
    pass


class Descriptor(namedtuple('Descriptor',
    ['inputs', 'outputs', 'params', 'categories'])):

    def serialize(self):
        res = ""
        res += "(" + ",".join(self.inputs) + ")"
        res += "(" + ",".join(self.outputs) + ")"
        paramStrs = []
        for type, name, defl in self.params:
          paramStrs.append(type + ":" + name + ":" + defl)
        res += "(" + ",".join(paramStrs) + ")"
        res += "(" + ",".join(self.categories) + ")"
        return res


class INode(abc.ABC):
    def __init__(self):
        self.__params = {}
        self.__inputs = {}

    def get_node_name(self):
        return getNodeName(self)

    def set_input(self, name, value):
        self.__inputs[name] = value

    def set_param(self, name, value):
        self.__params[name] = value

    def get_input(self, name):
        return self.__inputs[name]

    def get_param(self, name):
        return self.__params[name]

    def has_input(self, name):
        return name in self.__inputs

    def get_output(self, name):
        myname = self.get_node_name()
        return getObject(myname + "::" + name)

    def set_output(self, name, value):
        myname = self.get_node_name()
        setObject(myname + "::" + name, value)

    @abc.abstractmethod
    def apply(self):
        pass

    def on_apply(self):
        ok = True
        if self.has_input("COND"):
          cond = self.get_input("COND")
          ok = bool(cond)
        if ok:
          self.apply()
        self.set_output("DST", BooleanObject())


def isPyNodeType(type):
    return type in nodeClasses


def isPyNodeName(name):
    return name in nodes


def isPyObject(name):
    return name in objects


def addNode(type, name):
    node = nodeClasses[type]()
    nodesRev[node] = name
    nodes[name] = node


def setNodeParam(name, key, value):
    nodes[name].set_param(key, value)


def setNodeInput(name, key, srcname):
    obj = objects[srcname]
    nodes[name].set_input(key, obj)


def applyNode(name):
    nodes[name].on_apply()


def getNodeName(node):
    return nodesRev[node]


def setObject(name, object):
    objects[name] = object


def getObject(name):
    return objects[name]


def defNodeClassByCtor(ctor, name, desc):
    nodeClasses[name] = ctor
    nodeDescriptors[name] = desc


def defNodeClass(cls):
    name = getattr(cls, 'z_name', cls.__name__)

    def tostrlist(x):
        if isinstance(x, str):
            return [x]
        else:
            return list(x)

    inputs = tostrlist(getattr(cls, 'z_inputs', []))
    outputs = tostrlist(getattr(cls, 'z_outputs', []))
    params = list(ParamDescriptor(*x) for x in getattr(cls, 'z_params', []))
    categories = tostrlist(getattr(cls, 'z_categories', []))

    inputs.append("SRC")
    inputs.append("COND")
    outputs.append("DST")

    desc = Descriptor(inputs, outputs, params, categories)

    defNodeClassByCtor(cls, name, desc)


def dumpDescriptors():
    res = ""
    for name, desc in nodeDescriptors.items():
        res += name + ":" + desc.serialize() + "\n"
    return res


nodeDescriptors = {}
nodeClasses = {}
objects = {}
nodesRev = {}
nodes = {}


__all__ = [
    'INode',
    'IObject',
    'BooleanObject',
    'defNodeClass',
]
