#pragma once

/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <memory>

#include "zensim/Reflection.h"
#include "zensim/tpls/tl/function_ref.hpp"
#include "zensim/types/Function.h"
// #include <vsg/core/Allocator.h>
// #include <vsg/core/ConstVisitor.h>
// #include <vsg/core/Visitor.h>

// #include <vsg/traversals/RecordTraversal.h>

namespace zs {

  // Use the Curiously Recurring Template Pattern
  // to provide the classes versions of accept(..) and sizeofObject()
  template <typename ParentClass, typename Subclass> struct Inherit : public ParentClass {
  public:
    template <typename... Args> Inherit(Args &&...args)
        : ParentClass{std::forward<Args>(args)...} {}

    template <typename Allocator, typename... Args>
    static std::unique_ptr<Subclass> create(Allocator &&allocator, Args &&...args) {
      if (allocator) {
        // need to think about alignment...
        const std::size_t size = sizeof(Subclass);
        void *ptr = allocator->allocate(size);

        std::unique_ptr<Subclass> object(new (ptr) Subclass(allocator, args...));

        return object;
      } else
        return std::unique_ptr<Subclass>(new Subclass(args...));
    }

    template <typename... Args> static std::unique_ptr<Subclass> create(Args &&...args) {
      return std::unique_ptr<Subclass>(new Subclass(args...));
    }

    template <typename... Args>
    static std::unique_ptr<Subclass> create_if(bool flag, Args &&...args) {
      if (flag) return std::unique_ptr<Subclass>(new Subclass(args...));
      return {};
    }

    constexpr std::size_t sizeofObject() const noexcept { return sizeof(Subclass); }
    constexpr const char *className() const noexcept { return type_name<Subclass>(); }
    constexpr const std::type_info &typeInfo() const noexcept { return typeid(Subclass); }
    bool isCompatible(const std::type_info &type) const noexcept {
      /// recursive check
      return typeid(Subclass) == type ? true : ParentClass::isCompatible(type);
    }

    /// visitor should not be temporary
    void accept(tl::function_ref<void(Subclass &)> visitor) {
      visitor(static_cast<Subclass &>(*this));
    }
    void accept(tl::function_ref<void(const Subclass &)> visitor) const {
      visitor(static_cast<const Subclass &>(*this));
    }
  };

}  // namespace zs
