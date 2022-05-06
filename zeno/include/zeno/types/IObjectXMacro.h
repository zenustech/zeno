#pragma once

#define ZENO_XMACRO_IObject(PER, ...) \
    PER(PrimitiveObject, __VA_ARGS__) \
    PER(NumericObject, __VA_ARGS__) \
    PER(StringObject, __VA_ARGS__) \
    PER(CameraObject, __VA_ARGS__) \
    PER(LightObject, __VA_ARGS__) \
    PER(DummyObject, __VA_ARGS__)
