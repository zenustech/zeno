#pragma once


#define ZENO_DOP_DEFCLASS(Class, ...) \
    static int _zeno_dop_defclass_##Class = ( \
ZENO_NAMESPACE::dop::add_descriptor(#Class, std::make_unique<Class>, __VA_ARGS__), \
    1);


#define ZENO_DOP_INTERFACE(Class, ...) \
static int _zeno_dop_interface_##Class = ( \
    ZENO_NAMESPACE::dop::add_descriptor(#Class, std::make_unique<ZENO_NAMESPACE::dop::OverloadNode>, __VA_ARGS__), \
    1);


#define ZENO_DOP_IMPLEMENT(Class, func, ...) \
static int _zeno_dop_implement_##Class##_func_##func = ( \
    ZENO_NAMESPACE::dop::add_overloading(#Class, func, __VA_ARGS__), \
    1);


#define ZENO_DOP_DEFUN(Class, sig, ...) \
static int _zeno_dop_defun_##Class = ( \
    ZENO_NAMESPACE::dop::add_descriptor(#Class, std::make_unique<ZENO_NAMESPACE::dop::OverloadNode>, __VA_ARGS__), \
    ZENO_NAMESPACE::dop::add_overloading(#Class, Class, std::vector<std::type_index> sig), \
    1);
