#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {

template <class T> struct attr_to_vdb_type {};

template <> struct attr_to_vdb_type<float> {
  static constexpr bool is_scalar = true;
  using type = VDBFloatGrid;
};

template <> struct attr_to_vdb_type<vec3f> {
  static constexpr bool is_scalar = false;
  using type = VDBFloat3Grid;
};

template <> struct attr_to_vdb_type<int> {
  static constexpr bool is_scalar = true;
  using type = VDBIntGrid;
};

template <> struct attr_to_vdb_type<vec3i> {
  static constexpr bool is_scalar = false;
  using type = VDBInt3Grid;
};

template <class T>
void sampleVDBAttribute(std::vector<vec3f> const &pos, std::vector<T> &arr,
                        VDBGrid *ggrid) {
  using VDBType = typename attr_to_vdb_type<T>::type;
  auto ptr = dynamic_cast<VDBType *>(ggrid);
  if (!ptr) {
    printf("ERROR: vdb attribute type mismatch!\n");
    return;
  }
  auto grid = ptr->m_grid;

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    auto p0 = pos[i];
    auto p1 = vec_to_other<openvdb::Vec3R>(p0);
    auto p2 = grid->worldToIndex(p1);
    auto val = openvdb::tools::BoxSampler::sample(grid->tree(), p2);
    if constexpr (attr_to_vdb_type<T>::is_scalar) {
      arr[i] = val;
    } else {
      arr[i] = other_to_vec<3>(val);
    }
  }
}

struct SampleVDBToPrimitive : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto grid = get_input<VDBGrid>("vdbGrid");
    auto attr = get_input<StringObject>("primAttr")->get();
    auto sampleby = get_input<StringObject>("sampleBy")->get();
    auto &pos = prim->attr<vec3f>(sampleby);

    if (dynamic_cast<VDBFloatGrid *>(grid.get()))
        prim->add_attr<float>(attr);
    else if (dynamic_cast<VDBFloat3Grid *>(grid.get()))
        prim->add_attr<vec3f>(attr);
    else
        printf("unknown vdb grid type\n");

    std::visit([&](auto &vel) { sampleVDBAttribute(pos, vel, grid.get()); },
               prim->attr(attr));

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(SampleVDBToPrimitive, {
                                     {"prim", "vdbGrid", {"string", "sampleBy","pos"}, {"string", "primAttr", "sdf"}},
                                     {"prim"},
                                     {},
                                     {"openvdb"},
                                 });


} // namespace zeno
