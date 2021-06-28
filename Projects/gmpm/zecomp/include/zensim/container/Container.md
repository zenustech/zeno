**Vector**
==============
Similar to *std::vector*, *zs::Vector* is a container which stores elements contiguously in memory. Both adopt similar implementation guidelines, including rule of 5, RAII, etc.. But they are fundamentally different by design.

It (currently) adopts **umpire::strategy::MixedPool** as the go-to allocator for the memory management. There is no allocator handle within *Vector* object, and the underlying allocator is opaque to user. However, users can specify the **memory space** (CPU or GPU) and **processor id** (-1 for CPU and 0+ for GPU indices) during construction, and they should be careful about which **execution space** the vector object is accessed, e.g. access a vector element stored in GPU locally should not be accessed on the host side, unless it is unified memory.

The requirement of the element type is relatively more strict than std::vector, i.e. std::is_trivial<T> should hold true, to be able to complete memory operations on the host side, even though the underlying memory space is not. Only when std::is_trivially_copyable<T> holds true can one safely copy the vector elements around, even among different memory spaces and processors, no need to call elements' ctors or assignment operators.

# Type specification
```cpp
#include "zensim/container/Vector.hpp"
Vector<T> var{...};  //< T is the element type
```

# Construct
See Library/resource/Resource.h for info regarding memory space.
See Library/execution/ExecutionPolicy.h for info regarding processor id.
Copy constructors are exception safe.
Move constructors follow the guidelines.
See Library/container/Vector.hpp for more details.

```cpp
Vector<T> a{memsrc_e::um, -1};  ///< "default" ctor, stored on host
Vector<T> b{10, memsrc_e::um, 0};  ///< allocate an array of 10 elements within unified memory, preferred storage location is 0-th GPU 
Vector<T> c{b};  ///< a copy from b
Vector<T> d{std::move(b)};  ///< moved from b
```

# Iterator
```cpp
/// ranged-for way
for (auto &e : a)
    e...;
/// indexed-way
for (auto iter = a.begin(), ed = a.end(); iter != ed; ++iter)
    *iter...;
```

# Access
```cpp
a(3);   ///< access 3-th element (valid on both host side and device side)
a[4];   ///< access 4-th element (only valid on host side)
```

# resize, assignment, ...
```cpp
d.resize(20);
b = a;
c = std::move(a);
b.capacity();
b.size();
b.empty();
```

# insert, push_back, emplace_back
todo

**TileVector**
==============
TileVector resembles *Cabana* in that both adopt an aosoa-style layout. For particle and grid structure build.


**Hash Table**
==============
Unlike std::unordered_map, HashTable is a lot more . There are basically three types of hash tables.
1. for sparse matrix (small hash tables for rows/columns)
2. integer hash table
3. tuple hash table

Maintenance operations can only be called one at a time, e.g. there should not be more than one type of these opertions (insert, query or delete) under execution at the same time.
