#pragma once
#include <string_view>
#include <vector>
#include <zeno/core/INode.h>
#include <zeno/types/ListObject.h>
#include <zeno/zeno.h>

// only use this macro within a zeno::INode::apply()
#define RETRIEVE_OBJECT_PTRS(T, STR)                                           \
  ([this](const std::string_view str) {                                        \
    std::vector<T *> objPtrs{};                                                \
    if (has_input<T>(str.data()))                                              \
      objPtrs.push_back(get_input<T>(str.data()).get());                       \
    else if (has_input<zeno::ListObject>(str.data())) {                        \
      auto &objSharedPtrLists = *get_input<zeno::ListObject>(str.data());      \
      for (auto &&objSharedPtr : objSharedPtrLists.get())                      \
        if (auto ptr = dynamic_cast<T *>(objSharedPtr.get()); ptr != nullptr)  \
          objPtrs.push_back(ptr);                                              \
    }                                                                          \
    return objPtrs;                                                            \
  })(STR);

namespace zeno {} // namespace zeno