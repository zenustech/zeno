from .zeno import * 

@register_object_type("zsVec")
class ZsSmallVec(ZenoObject):
    def __repr__(self) -> str:
        return '[zs small vec at {}]'.format(self._handle)

    def asObject(self):
        return ZenoObject.fromHandle(self._handle)
    