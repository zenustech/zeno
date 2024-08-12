#include "Vector.hpp"
#include "zensim/ZpcFunctional.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <fmt/core.h>
#include <tuple>
#include <variant>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>

namespace zeno {
struct MakeZsVector : INode {
  void apply() override {
    // TODO
    auto input_size = get_input2<int>("size");
    auto input_memsrc = get_input2<std::string>("memsrc");
    auto intput_devid = get_input2<int>("dev_id");
    // auto input_virtual = get_input2<bool>("virtual");
    auto intput_elem_type = get_input2<std::string>("elem_type");

    zs::memsrc_e memsrc;
    if (input_memsrc == "host")
      memsrc = zs::memsrc_e::host;
    else if (input_memsrc == "device")
      memsrc = zs::memsrc_e::device;
    else
      memsrc = zs::memsrc_e::um;

#define MAKE_VECTOR_OBJ_T(T)                                                   \
  if (intput_elem_type == #T) {                                                \
    auto allocator =                                                           \
        zs::get_memory_source(memsrc, static_cast<zs::ProcID>(intput_devid));  \
    vectorObj->set(zs::Vector<T, zs::ZSPmrAllocator<false>>{allocator, 0});    \
  }

    auto vectorObj = std::make_shared<ZsVectorObject>();
    MAKE_VECTOR_OBJ_T(int)
    MAKE_VECTOR_OBJ_T(float)
    MAKE_VECTOR_OBJ_T(double)
    std::visit([input_size](auto &vec) { vec.resize(input_size); },
               vectorObj->value);

    set_output("ZsVector", std::move(vectorObj));
  }
};

//  memsrc, size, elem_type, dev_id, virtual
ZENDEFNODE(MakeZsVector, {
                             {{"int", "size", "0"},
                              {"enum host device um", "memsrc", "device"},
                              {"int", "dev_id", "0"},
                              //   {"bool", "virtual", "false"},
                              {"enum float int double", "elem_type", "float"}},
                             {"ZsVector"},
                             {},
                             {"PyZFX"},
                         });

struct ReduceZsVector : INode {
  void apply() override {
    auto vectorObj = get_input<ZsVectorObject>("ZsVector");
    auto opStr = get_input2<std::string>("op");
    auto &vector = vectorObj->value;

    float result;
    std::visit(
        [&result, &opStr](auto &vector) {
          auto pol = zs::cuda_exec();
          using vector_t = RM_CVREF_T(vector);
          using val_t = typename vector_t::value_type;
          zs::Vector<val_t> res{1, zs::memsrc_e::device};
          if (opStr == "add")
            zs::reduce(pol, std::begin(vector), std::end(vector),
                       std::begin(res), static_cast<val_t>(0),
                       zs::plus<val_t>{});
          else if (opStr == "max")
            zs::reduce(pol, std::begin(vector), std::end(vector),
                       std::begin(res), zs::limits<val_t>::min(),
                       zs::getmax<val_t>{});
          else
            zs::reduce(pol, std::begin(vector), std::end(vector),
                       std::begin(res), zs::limits<val_t>::max(),
                       zs::getmin<val_t>{});
          result = static_cast<float>(res.getVal());
        },
        vector);
    set_output2("result", result);
  }
};

ZENDEFNODE(ReduceZsVector, {
                               {"ZsVector", {"enum add max min", "op", "add"}},
                               {"result"},
                               {},
                               {"PyZFX"},
                           });

template <typename T>
struct _is_float : std::bool_constant<zs::is_floating_point_v<T>> {};
template <size_t N, typename T>
struct _is_float<zeno::vec<N, T>>
    : std::bool_constant<zs::is_floating_point_v<T>> {};

struct CopyZsVectorTo : INode {
  void apply() override {
    auto vectorObj = get_input<ZsVectorObject>("ZsVector");
    auto prim = get_input<PrimitiveObject>("prim");
    auto attr = get_input2<std::string>("attr");
    auto &vector = vectorObj->value;

    std::visit(
        [&prim, &attr](auto &vector) {
          using vector_t = RM_CVREF_T(vector);
          using val_t = typename vector_t::value_type;
          if constexpr (zs::is_same_v<val_t, float> ||
                        zs::is_same_v<val_t, int>) {

            auto process = [&prim = prim, &vector = vector,
                            &attr](auto &primAttr) {
              using T = RM_CVREF_T(primAttr[0]);
              constexpr bool sameType =
                  _is_float<T>::value == zs::is_same_v<val_t, float>;

              constexpr auto nbytes = sizeof(T);
              if (prim->size() * (nbytes / sizeof(float)) < vector.size()) {
                fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                if (vector.size() % (nbytes / sizeof(float)) != 0) {
                  throw std::runtime_error(fmt::format(
                      "vector of type {} copied to primattr [{}] "
                      "containing {} "
                      "elements of type {}, yet vector size is {}\n",
                      zs::get_type_str<val_t>().asChars(), attr, prim->size(),
                      zs::get_type_str<T>().asChars(), vector.size()));
                }
                /// @note this does not invalidate primAttr
                prim->resize(vector.size() / (nbytes / sizeof(float)));
              }
              if constexpr (sameType)
                zs::Resource::copy(
                    zs::MemoryEntity{zs::MemoryLocation{zs::memsrc_e::host, -1},
                                     (void *)primAttr.data()},
                    zs::MemoryEntity{vector.memoryLocation(),
                                     (void *)vector.data()},
                    sizeof(val_t) * vector.size());
              else {
                if constexpr (zs::is_same_v<val_t, float>) {
                  // float -> int
                  zs::omp_exec()(
                      zs::range(vector.size()),
                      [&vector, primAttrAddr = (int *)primAttr.data()](
                          size_t i) { primAttrAddr[i] = (int)vector[i]; });
                } else {
                  // int -> float
                  zs::omp_exec()(
                      zs::range(vector.size()),
                      [&vector, primAttrAddr = (float *)primAttr.data()](
                          size_t i) { primAttrAddr[i] = (float)vector[i]; });
                }
              }
            };
            if (attr == "pos")
              process(prim->verts.values);
            else
              std::visit(process, prim->attr(attr));

          } else if constexpr (zs::is_same_v<val_t, double>) {
            auto process = [&prim = prim, &vector = vector,
                            &attr](auto &primAttr) {
              using T = RM_CVREF_T(primAttr[0]);
              constexpr auto nbytes = sizeof(T);
              if (prim->size() * (nbytes / sizeof(float)) < vector.size()) {
                fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                if (vector.size() % (nbytes / sizeof(float)) != 0) {
                  throw std::runtime_error(fmt::format(
                      "vector of type {} copied to primattr [{}] "
                      "containing {} "
                      "elements of type {}, yet vector size is {}\n",
                      zs::get_type_str<val_t>().asChars(), attr, prim->size(),
                      zs::get_type_str<T>().asChars(), vector.size()));
                }
                /// @note this does not invalidate primAttr
                prim->resize(vector.size() / (nbytes / sizeof(float)));
              }

              if constexpr (!_is_float<T>::value) {
                // double -> int
                zs::omp_exec()(
                    zs::range(vector.size()),
                    [&vector, primAttrAddr = (int *)primAttr.data()](size_t i) {
                      primAttrAddr[i] = (int)vector[i];
                    });
              } else {
                // double -> float
                zs::omp_exec()(
                    zs::range(vector.size()),
                    [&vector, primAttrAddr = (float *)primAttr.data()](
                        size_t i) { primAttrAddr[i] = (float)vector[i]; });
              }
            };

            if (attr == "pos")
              process(prim->verts.values);
            else
              std::visit(process, prim->attr(attr));
          }
        },
        vector);

    set_output2("prim", prim);
  }
};

ZENDEFNODE(CopyZsVectorTo, {
                               {"ZsVector", "prim", {"string", "attr", "clr"}},
                               {"prim"},
                               {},
                               {"PyZFX"},
                           });

struct CopyZsVectorFrom : INode {
  void apply() override {
    auto vectorObj = get_input<ZsVectorObject>("ZsVector");
    auto prim = get_input<PrimitiveObject>("prim");
    auto attr = get_input2<std::string>("attr");
    auto &vector = vectorObj->value;

    std::visit(
        [&prim, &attr](auto &vector) {
          using vector_t = RM_CVREF_T(vector);
          using val_t = typename vector_t::value_type;
          if constexpr (zs::is_same_v<val_t, float> ||
                        zs::is_same_v<val_t, int>) {

            auto process = [&prim = prim, &vector = vector,
                            &attr](auto &primAttr) {
              using T = RM_CVREF_T(primAttr[0]);
              constexpr bool sameType =
                  _is_float<T>::value == zs::is_same_v<val_t, float>;

              constexpr auto nbytes = sizeof(T);
              if (prim->size() * (nbytes / sizeof(float)) > vector.size()) {
                fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                vector.resize(prim->size() * (nbytes / sizeof(float)));
              }
              if constexpr (sameType)
                zs::Resource::copy(
                    zs::MemoryEntity{vector.memoryLocation(),
                                     (void *)vector.data()},
                    zs::MemoryEntity{zs::MemoryLocation{zs::memsrc_e::host, -1},
                                     (void *)primAttr.data()},
                    nbytes * prim->size());
              else {
                if constexpr (zs::is_same_v<val_t, float>) {
                  // float <- int
                  zs::omp_exec()(
                      zs::range(prim->size() * (nbytes / sizeof(float))),
                      [&vector, primAttrAddr = (int *)primAttr.data()](
                          size_t i) { vector[i] = (float)primAttrAddr[i]; });
                } else {
                  // int <- float
                  zs::omp_exec()(
                      zs::range(prim->size() * (nbytes / sizeof(float))),
                      [&vector, primAttrAddr = (float *)primAttr.data()](
                          size_t i) { vector[i] = (int)primAttrAddr[i]; });
                }
              }
            };

            if (attr == "pos")
              process(prim->verts.values);
            else
              std::visit(process, prim->attr(attr));

          } else if constexpr (zs::is_same_v<val_t, double>) {
            auto process = [&prim = prim, &vector = vector,
                            &attr](auto &primAttr) {
              using T = RM_CVREF_T(primAttr[0]);
              constexpr auto nbytes = sizeof(T);
              if (prim->size() * (nbytes / sizeof(float)) > vector.size()) {
                fmt::print("BEWARE! copy sizes mismatch! resize to match.\n");
                vector.resize(prim->size() * (nbytes / sizeof(float)));
              }

              if constexpr (!_is_float<T>::value) {
                // double <- int
                zs::omp_exec()(
                    zs::range(prim->size() * (nbytes / sizeof(float))),
                    [&vector, primAttrAddr = (int *)primAttr.data()](size_t i) {
                      vector[i] = (double)primAttrAddr[i];
                    });
              } else {
                // double <- float
                zs::omp_exec()(
                    zs::range(prim->size() * (nbytes / sizeof(float))),
                    [&vector, primAttrAddr = (float *)primAttr.data()](
                        size_t i) { vector[i] = (double)primAttrAddr[i]; });
              }
            };

            if (attr == "pos")
              process(prim->verts.values);
            else
              std::visit(process, prim->attr(attr));
          }
        },
        vector);

    set_output2("ZsVector", vectorObj);
  }
};

ZENDEFNODE(CopyZsVectorFrom,
           {
               {"ZsVector", "prim", {"string", "attr", "clr"}},
               {"ZsVector"},
               {},
               {"PyZFX"},
           });
} // namespace zeno