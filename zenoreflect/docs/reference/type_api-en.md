# Reflection Interface

In general, there are two interfaces provided to directly obtain type information using the `typename` expression:

```cpp
// Get generated RTTI information
zeno::reflect::type_info<T>()

// Get reflection type information
zeno::reflect::get_type<T>()
```

## RTTI Information

It is well known that C++ Runtime Type Information (RTTI) does not guarantee ABI compatibility between binaries. This is understandable because it is usually generated at compile time and placed in the read-only data segment (though it doesn't necessarily have to be in that segment). Also, C++ does not require globally unique type names, so compilers generate unique names and hash values within each translation unit.

Therefore, we generate compile-time accessible RTTI information for reflected types and the types used in reflected types. This information is guaranteed to be unique across any platform, provided the limitations of this project are not violated.

This information can be accessed via the following template function. Before using this function, you need to include the generated header file `#include "reflect/reflection.generated.hpp"` so that the compiler can correctly obtain the type template specialization.

```cpp
zeno::reflect::type_info<T>()
```

## Reflection Information

As for the `get_type<T>()` interface, it returns a `TypeHandle` object, which is a type information handle with low copy overhead.

Given that it is **ensured** to be a type with a reflection marker, you can directly use the `->` operator to access the `TypeBase` interface.
Even if it is not a reflected type, type comparison can still be performed.

## More Direct Reflection Information

Since this is runtime reflection, it also supports obtaining reflection information from the type name. However, these interfaces related to the type registry are not yet stable and may change at any time.

You can refer to the `crates/libreflect/include/reflect/registry.hpp` file for more details.
