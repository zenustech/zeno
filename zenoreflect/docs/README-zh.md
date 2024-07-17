# 使用说明

## 目录

[Any容器 / 类型擦除容器](reference/any-zh.md)

[反射接口](reference/type_api-zh.md)

## 快速上手

1. 把本项目作为子模块添加到你的项目中
2. 引入本项目作
```cmake
add_subdirectory(/path/to/ZenoReflect)
```
3. 然后在你需要反射的模块中使用（参考`examples`文件夹下内容）
```cmake
make_absolute_paths(REFLECTION_HEADERS 
    include/data.h
    include/test.h
) 
zeno_declare_reflection_support(ReflectExample "${REFLECTION_HEADERS}")
```

## 机制

首先要明确的是，这是一个运行时反射。在执行一些操作时，会进行类型擦除以满足运行时接口的要求。
你可以在运行时使用任何类型（包括被实例化的模版），只要它们被正确地注册到反射系统中。

还有，目前类型擦除系统还未完善，可能会出现问题。

<del>目前暂不支持类型的手动注册</del>，所有的注册代码都是在编译时根据**反射标记**自动生成的。

现在可以在被注册扫描的头文件中使用`REFLECT_REGISTER_RTTI_TYPE_MANUAL`宏来把任意类型注册进RTTI系统中。

调用`zeno_declare_reflection_support`会自动产生一个进行反射生成的target，并且你的target会依赖这个新的target。
这意味着反射信息只会保证在你的target编译前被生成，不要在其它任何未启用反射的target中假设静态反射信息已经存在。当然在任何地方使用运行时的API是安全的（除了静态初始化阶段，不去假设静态初始化顺序是一个好习惯）。

target的运行时信息注册会通过添加一个由反射生成器生成的源码文件来实现，它当前会位于`[CMAKE文件夹]/intermediate/[target名称]/[target名称].generated.cpp`。

而所需的静态信息则会生成在`crates/libgenerated/include/reflect`文件夹中。如果你需要静态反射信息，你要在你代码中写上`#include "reflect/reflection.generated.hpp"`。在你为你的target启用反射时，`libgenerated`就会添加为你target的`interface`类型依赖。

## FQA

> Q: 有什么限制
> A: 不能有同名类型同时拥有反射标记，且头文件名不应重名

