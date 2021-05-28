'''
Python APIs
'''


import abc
from collections import namedtuple

from .api import requireObject


class IObject:
    pass


class BooleanObject(IObject):
    def __init__(self, value=True):
        self.__value = bool(value)

    def __bool__(self):
        return self.__value


class ParamDescriptor(namedtuple('ParamDescriptor', 'type, name, defl')):
    pass


class Descriptor(namedtuple('Descriptor',
    'inputs, outputs, params, categories')):
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

    def set_input_ref(self, name, srcname):
        self.__inputs[name] = srcname

    def set_param(self, name, value):
        self.__params[name] = value

    def get_input_ref(self, name):
        return self.__inputs[name]

    def get_input(self, name):
        ref = self.get_input_ref(name)
        requireObject(ref)
        return getObject(ref)

    def get_param(self, name):
        return self.__params[name]

    def has_input(self, name):
        return name in self.__inputs

    def get_output_ref(self, name):
        myname = self.get_node_name()
        return myname + "::" + name

    def get_output(self, name):
        ref = self.get_output_ref(name)
        return getObject(ref)

    def set_output(self, name, value):
        ref = self.get_output_ref(name)
        setObject(ref, value)

    def set_output_ref(self, name, srcname):
        ref = self.get_output_ref(name)
        requireObject(srcname)
        setReference(ref, srcname)

    def init(self):
        pass

    def on_init(self):
        self.init()

    @abc.abstractmethod
    def apply(self):
        pass

    def on_apply(self):
        if self.has_input("SRC"):
          self.get_input("SRC")  # to invoke requireObject
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
    if name in nodes:
        return
    node = nodeClasses[type]()
    nodesRev[node] = name
    nodes[name] = node


def initNode(name):
    node = nodes[name]
    node.on_init()


def setNodeParam(name, key, value):
    nodes[name].set_param(key, value)


def setNodeInput(name, key, srcname):
    nodes[name].set_input_ref(key, srcname)


def applyNode(name):
    nodes[name].on_apply()


def getNodeName(node):
    return nodesRev[node]


def setObject(name, object):
    objects[name] = object


def setReference(name, srcname):
    objects[name] = objects[srcname]


def getObject(name):
    return objects[name]


def getNodeByName(name):
    return nodes[name]


def defNodeClassByCtor(ctor, name, desc):
    nodeClasses[name] = ctor
    nodeDescriptors[name] = desc


def defNodeClass(cls):
    cls.z_name = name = getattr(cls, 'z_name', cls.__name__)

    def tostrlist(x):
        if isinstance(x, str):
            return [x]
        else:
            return list(x)

    cls.z_inputs = inputs = tostrlist(getattr(cls, 'z_inputs', []))
    cls.z_outputs = outputs = tostrlist(getattr(cls, 'z_outputs', []))
    cls.z_params = params = [ParamDescriptor(*x) for x in getattr(cls, 'z_params', [])]
    cls.z_categories = categories = tostrlist(getattr(cls, 'z_categories', []))

    inputs.append("SRC")
    inputs.append("COND")
    outputs.append("DST")

    desc = Descriptor(inputs, outputs, params, categories)
    defNodeClassByCtor(cls, name, desc)
    return cls


def dumpDescriptors():
    res = ""
    for name, desc in nodeDescriptors.items():
        res += name + ":" + desc.serialize() + "\n"
    return res


nodeDescriptors = {}
nodeClasses = {}
objects = {}
nodes = {}
nodesRev = {}


__all__ = [
    'INode',
    'IObject',
    'BooleanObject',
    'defNodeClass',
]
