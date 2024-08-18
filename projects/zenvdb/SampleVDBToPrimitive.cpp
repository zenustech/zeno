#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/UserData.h>
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

template <typename T, typename = void>
struct is_vdb_to_prim_convertible {
  static constexpr bool value = false;
};
template <typename T>
struct is_vdb_to_prim_convertible<std::vector<T>, std::void_t<typename attr_to_vdb_type<T>::type>> {
  static constexpr bool value = true;
};

template <class T>
void sampleVDBAttribute(std::vector<vec3f> const &pos, std::vector<T> &arr,
                        VDBGrid *ggrid) {
  using VDBType = typename attr_to_vdb_type<T>::type;
  auto ptr = dynamic_cast<VDBType *>(ggrid);
  if (!ptr) {
    zeno::log_error("ERROR: vdb attribute type mismatch!");
    throw std::runtime_error("ERROR: vdb attribute type mismatch!");
  }
  auto grid = ptr->m_grid;

#pragma omp parallel for
  for (int i = 0; i < pos.size(); i++) {
    auto p0 = pos[i];
    auto p1 = vec_to_other<openvdb::Vec3R>(p0);
    auto p2 = grid->worldToIndex(p1);
    auto val = openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), p2);
    if constexpr (attr_to_vdb_type<T>::is_scalar) {
      arr[i] = val;
    } else {
      arr[i] = other_to_vec<3>(val);
    }
  }
}
template <class T>
void sampleVDBAttribute2(
        std::vector<vec3f> const &pos,
        std::vector<T> &arr,
        VDBGrid *ggrid,
        float remapMin,
        float remapMax
) {
    using VDBType = typename attr_to_vdb_type<T>::type;
    auto ptr = dynamic_cast<VDBType *>(ggrid);
    if (!ptr) {
        zeno::log_error("ERROR: vdb attribute type mismatch!");
        throw std::runtime_error("ERROR: vdb attribute type mismatch!");
    }
    auto grid = ptr->m_grid;

    #pragma omp parallel for
    for (int i = 0; i < pos.size(); i++) {
        auto p0 = (pos[i] - remapMin) / (remapMax - remapMin);
        auto p1 = vec_to_other<openvdb::Vec3R>(p0);
        auto p2 = grid->worldToIndex(p1);
        auto val = openvdb::tools::BoxSampler::sample(grid->getConstUnsafeAccessor(), p2);
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
    auto type = get_param<std::string>(("SampleType"));


    if (dynamic_cast<VDBFloatGrid *>(grid.get()))
        prim->add_attr<float>(attr);
    else if (dynamic_cast<VDBFloat3Grid *>(grid.get()))
        prim->add_attr<vec3f>(attr);
    else
        throw zeno::Exception("unknown vdb grid type\n");

    if(type == "Periodic")
    {
      //TODO: if the sample VDB is considered periodic, 
      //fill the boundary values
      //warp the sample coordinates
    }

    //std::visit([&](auto &vel) { 
    prim->attr_visit(attr, [&] (auto &vel) {
      if constexpr (is_vdb_to_prim_convertible<std::decay_t<decltype(vel)>>::value)
        sampleVDBAttribute(pos, vel, grid.get()); 
    });
               //prim->attr(attr));


    if(type == "Periodic")
    {
      //TODO: if the sample VDB is considered periodic, 
      //set back boundary values to be zero
    }

    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(SampleVDBToPrimitive, {
                                     {
                                        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
                                        {"VDBGrid", "vdbGrid", "", zeno::Socket_ReadOnly},
                                        {gParamType_String, "sampleBy","pos"},
                                        {gParamType_String, "primAttr", "sdf"}},
                                     {gParamType_Primitive, "prim"},
                                     {{"enum Clamp Periodic", "SampleType", "Clamp"}},
                                     {"openvdb"},
                                 });

static void primSampleVDB(
        std::shared_ptr<PrimitiveObject> prim,
        const std::string &srcChannel,
        const std::string &dstChannel,
        std::shared_ptr<VDBGrid> grid,
        float remapMin,
        float remapMax
) {
    auto &pos = prim->attr<vec3f>(srcChannel);
    if (dynamic_cast<VDBFloatGrid *>(grid.get())) {
        prim->add_attr<float>(dstChannel);
    }
    else if (dynamic_cast<VDBFloat3Grid *>(grid.get())) {
        prim->add_attr<vec3f>(dstChannel);
    }
    else {
        throw std::runtime_error("unknown vdb grid type");
    }
    prim->attr_visit(dstChannel, [&] (auto &vel) {
        if constexpr (is_vdb_to_prim_convertible<std::decay_t<decltype(vel)>>::value)
            sampleVDBAttribute2(pos, vel, grid.get(), remapMin, remapMax);
    });
}

struct PrimSample3D : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto grid = get_input<VDBGrid>("vdbGrid");
        auto dstChannel = get_input2<std::string>("dstChannel");
        auto srcChannel = get_input2<std::string>("srcChannel");
        auto remapMin = get_input2<float>("remapMin");
        auto remapMax = get_input2<float>("remapMax");

        primSampleVDB(prim, srcChannel, dstChannel, grid, remapMin, remapMax);
        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample3D, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {"VDBGrid", "vdbGrid", "", zeno::Socket_ReadOnly},
        {gParamType_String, "srcChannel", "pos"},
        {gParamType_String, "dstChannel", "clr"},
        {gParamType_Float, "remapMin", "0"},
        {gParamType_Float, "remapMax", "1"},
    },
    {
        {gParamType_Primitive, "outPrim"}
    },
    {},
    {"primitive"},
});
struct PrimSample : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto srcChannel = get_input2<std::string>("srcChannel");
        auto dstChannel = get_input2<std::string>("dstChannel");
        auto remapMin = get_input2<float>("remapMin");
        auto remapMax = get_input2<float>("remapMax");
        auto wrap = get_input2<std::string>("wrap");
        auto borderColor = get_input2<vec3f>("borderColor");
        if (has_input<PrimitiveObject>("sampledObject") && get_input<PrimitiveObject>("sampledObject")->userData().has("isImage")) {
            auto image = get_input<PrimitiveObject>("sampledObject");
            primSampleTexture(prim, srcChannel, "vertex", dstChannel, image, wrap, borderColor, remapMin, remapMax);
        }
        else if (has_input<HeatmapObject>("sampledObject")) {
            auto heatmap = get_input<HeatmapObject>("sampledObject");
            primSampleHeatmap(prim, srcChannel, dstChannel, heatmap, remapMin, remapMax);
        }
        else if (has_input<VDBGrid>("sampledObject")) {
            auto grid = get_input<VDBGrid>("vdbGrid");
            primSampleVDB(prim, srcChannel, dstChannel, grid, remapMin, remapMax);
        } else {
            throw zeno::Exception("unknown input type of sampledObject");
        }

        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {"object", "sampledObject", "", zeno::Socket_ReadOnly},
        {gParamType_String, "srcChannel", "uv"},
        {gParamType_String, "dstChannel", "clr"},
        {gParamType_Float, "remapMin", "0"},
        {gParamType_Float, "remapMax", "1"},
        {"enum REPEAT CLAMP_TO_EDGE CLAMP_TO_BORDER", "wrap", "REPEAT"},
        {gParamType_Vec3f, "borderColor", "0,0,0"},
    },
    {
        {gParamType_Primitive, "outPrim"}
    },
    {},
    {"primitive"},
});
} // namespace zeno
