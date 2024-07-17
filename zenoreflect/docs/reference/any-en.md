# Any

The `Any` type provides generic object operations after type erasure. It represents an object of any type but **cannot** represent references and pointers.

In other words, `Any` holds the value of an object, not its memory address. Therefore, consider the following examples:

```cpp
Foo foo;

// If moveable, the move constructor will be called. Otherwise, the copy constructor will be called.
Any any(zeno::reflect::move(foo));

// The copy constructor will always be called.
Any any(foo);
```

Note that in the examples above, if the type cannot be copy-constructed or move-constructed, or if there is a compatible implicit conversion constructor, it will result in a compilation error.

Therefore, we provide a `make_any` method to construct an `Any` object **in-place**.

```cpp
// Assume Foo has a constructor with three int parameters
Any any = zeno::reflect::make_any<Foo>(123, 456, 789);
```

## Move and Copy

As mentioned above, if `Any` is passed by value:

```cpp
Any origin_any = make_any<Foo>();

// This triggers the copy constructor to clone the internal value.
// If copy construction is not possible, an exception will be thrown.
Any duplicated_any = origin_any;
```

If you do not want to trigger the clone operation, you need to pass it by pointer, reference, or using move semantics.
