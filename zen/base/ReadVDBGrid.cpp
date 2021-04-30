#include <zen/VDBGrid.h>
#include <zen/zen.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
// openvdb::io::File(filename).write({grid});

namespace zenbase {

struct ReadVDBGrid : zen::INode {
  virtual void apply() override {
    auto path = std::get<std::string>(get_param("path"));
    auto type = std::get<std::string>(get_param("type"));
    std::unique_ptr<VDBGrid> data;
    if (type == "float") {
      data = zen::IObject::make<VDBFloatGrid>();
    } else if (type == "float3") {
      data = zen::IObject::make<VDBFloat3Grid>();
    } else if (type == "int") {
      data = zen::IObject::make<VDBIntGrid>();
    } else if (type == "int3") {
      data = zen::IObject::make<VDBInt3Grid>();
    } else if (type == "points") {
      data = zen::IObject::make<VDBPointsGrid>();
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    data->input(path);
    set_output("data", data);
  }
};
static int defReadVDBGrid = zen::defNodeClass<ReadVDBGrid>(
    "ReadVDBGrid", {/* inputs: */ {}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"string", "type", "float"},
                        {"string", "path", ""},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

template <class T> struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};
template <class T> using remove_cvref_t = typename remove_cvref<T>::type;
/// https://github.com/SuperV1234/ndctechtown2020/blob/master/7_a_match.pdf
template <typename... Fs> struct overload_set : Fs... {
  template <typename... Xs>
  constexpr overload_set(Xs &&...xs) : Fs{std::forward<Xs>(xs)}... {}
  using Fs::operator()...;
};
/// class template argument deduction
template <typename... Xs>
overload_set(Xs &&...xs) -> overload_set<remove_cvref_t<Xs>...>;

template <typename... Fs> constexpr auto make_overload_set(Fs &&...fs) {
  return overload_set<std::decay_t<Fs>...>(std::forward<Fs>(fs)...);
}

template <typename... Ts> using variant = std::variant<Ts...>;

template <typename... Fs> constexpr auto match(Fs &&...fs) {
  return [visitor = make_overload_set(std::forward<Fs>(fs)...)](
             auto &&...vs) -> decltype(auto) {
    return std::visit(visitor, std::forward<decltype(vs)>(vs)...);
  };
}

struct MakeVDBGrid : zen::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    auto type = std::get<std::string>(get_param("type"));
    auto structure = std::get<std::string>(get_param("structure"));
    auto name = std::get<std::string>(get_param("name"));
    std::unique_ptr<VDBGrid> data;
    // using RetT =
    //     std::variant<std::monostate, std::unique_ptr<VDBFloatGrid>,
    //                  std::unique_ptr<VDBFloat3Grid>,
    //                  std::unique_ptr<VDBIntGrid>, std::unique_ptr<VDBInt3Grid>,
    //                  std::unique_ptr<VDBPointsGrid>>;
    //RetT tmp;
    if (type == "float") {
      auto tmp = zen::IObject::make<VDBFloatGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "float3") {
      auto tmp = zen::IObject::make<VDBFloat3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      if (structure == "Staggered") {
        tmp->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
      }
      data = std::move(tmp);
    } else if (type == "int") {
      auto tmp = zen::IObject::make<VDBIntGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "int3") {
      auto tmp = zen::IObject::make<VDBInt3Grid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else if (type == "points") {
      auto tmp = zen::IObject::make<VDBPointsGrid>();
      tmp->m_grid->setTransform(openvdb::math::Transform::createLinearTransform(dx));
      tmp->m_grid->setName(name);
      data = std::move(tmp);
    } else {
      printf("%s\n", type.c_str());
      assert(0 && "bad VDBGrid type");
    }
    // match(
    //     [dx, &structure, &name](auto &ptr) {
    //       ptr->m_grid->setTransform(
    //           openvdb::math::Transform::createLinearTransform(dx));
    //       if constexpr (std::is_same_v<remove_cvref_t<decltype(ptr)>,
    //                                    std::unique_ptr<VDBFloat3Grid>>)
    //         if (structure == "Staggered") {
    //           ptr->m_grid->setGridClass(openvdb::GridClass::GRID_STAGGERED);
    //         }
    //       ptr->m_grid->setName(name);
    //     },
    //     [](std::monostate) {}, [](...) {})(tmp);
    // match([&data](auto &ptr) { data = std::move(ptr); }, [](std::monostate) {},
    //       [](...) {})(tmp);
    set_output("data", data);
  }
};

static int defMakeVDBGrid = zen::defNodeClass<MakeVDBGrid>(
    "MakeVDBGrid", {/* inputs: */ {}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                        {"float", "dx", "0.01"},
                        {"string", "type", "float"},
                        {"string", "structure", "Centered"},
                        {"string", "name", "Rename!"},
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

} // namespace zenbase
