#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/vec.h>
#include <zeno/zeno.h>
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
    auto attr = get_param<std::string>("primAttr");
    auto &pos = prim->attr<vec3f>("pos");
    std::visit([&](auto &vel) { sampleVDBAttribute(pos, vel, grid.get()); },
               prim->attr(attr));

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(SampleVDBToPrimitive, {
                                     {"prim", "vdbGrid"},
                                     {"prim"},
                                     {{"string", "primAttr", "rho"}},
                                     {"visualize"},
                                 });

struct GetVDBBound : INode {
  virtual void apply() override {
    auto grid = get_input<VDBGrid>("vdbGrid");
    auto bbmin = zeno::IObject::make<zeno::NumericObject>();
    auto bbmax = zeno::IObject::make<zeno::NumericObject>();

    zeno::vec3f bmin, bmax;
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    auto corner = box.min();
    auto length = box.max() - box.min();
    auto world_min = grid->indexToWorld(box.min());
    auto world_max = grid->indexToWorld(box.max());

    for (size_t d = 0; d < 3; d++) {
      bmin[d] = world_min[d];
      bmax[d] = world_max[d];
    }

    for (int dx = 0; dx < 2; dx++)
      for (int dy = 0; dy < 2; dy++)
        for (int dz = 0; dz < 2; dz++) {
          auto coord =
              corner + decltype(length){dx ? length[0] : 0, dy ? length[1] : 0,
                                        dz ? length[2] : 0};

          auto pos = grid->indexToWorld(coord);

          for (int d = 0; d < 3; d++) {
            bmin[d] = pos[d] < bmin[d] ? pos[d] : bmin[d];
            bmax[d] = pos[d] > bmax[d] ? pos[d] : bmax[d];
          }
        }

    bbmin->set<zeno::vec3f>(bmin);
    bbmax->set<zeno::vec3f>(bmax);
    set_output("bmin", bbmin);
    set_output("bmax", bbmax);
  }
};

ZENDEFNODE(GetVDBBound, {
                            {"vdbGrid"},
                            {"bmin", "bmax"},
                            {},
                            {"openvdb"},
                        });

// this is to be deprecated
struct HeatMap {
  std::vector<zeno::vec3f> colorBar;

  HeatMap() {
    colorBar.resize(1024);
    for (int i = 0; i < 512; i++) {
      colorBar[i] = (1.0f - (float)i / 512.0f) * zeno::vec3f(0, 0, 1) +
                    (float)i / 512.0f * zeno::vec3f(0, 1, 0);
      colorBar[i + 512] = (1.0f - (float)i / 512.0f) * zeno::vec3f(0, 1, 0) +
                          (float)i / 512.0f * zeno::vec3f(1, 0, 0);
    }
  }
  zeno::vec3f sample(float c) {
    int i = c * 1024.0f;
    int j = i + 1;
    zeno::clamp(i, 0, 1023);
    zeno::clamp(j, 0, 1023);
    float dx = c * 1024.0f - (float)i;
    zeno::clamp(dx, 0.0f, 1.0f);
    return colorBar[i] * (1.0f - dx) + dx * colorBar[j];
  }
};
static HeatMap tempHeatMap;

template <class T>
void colorFromAttr(std::vector<T> &arr, std::vector<zeno::vec3f> &clr,
                   float minval, float maxval, HeatMap *hm) {

#pragma omp parallel for
  for (int i = 0; i < arr.size(); i++) {
    float d;
    if constexpr (!std::is_same<T, float>::value)
      d = std::sqrt(zeno::dot(arr[i], arr[i]));
    else if constexpr (std::is_same<T, float>::value)
      d = arr[i];
    auto c =
        (std::max(std::min(d, maxval), minval) - minval) / (maxval - minval);
    clr[i] = hm->sample(c);
  }
}

struct HeatMapPrimitive : INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    std::string attrs = get_input<StringObject>("primdAttr")->get();
    auto minval = get_input<NumericObject>("min")->get<float>();
    auto maxval = get_input<NumericObject>("max")->get<float>();
    prim->add_attr<zeno::vec3f>("clr");

    std::visit(
        [&](auto &vel) {
          colorFromAttr(vel, prim->attr<zeno::vec3f>("clr"), minval, maxval,
                        &tempHeatMap);
        },
        prim->attr(attrs));
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(HeatMapPrimitive, {
                                 {"prim", "primdAttr", "min", "max"},
                                 {"prim"},
                                 {},
                                 {"visualize"},
                             });

} // namespace zeno
