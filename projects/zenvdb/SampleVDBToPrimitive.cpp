#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/types/HeatmapObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/UserData.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>
#include <zeno/geo/commonutil.h>

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
    auto prim = clone_input_PrimitiveObject("prim");
    auto grid = safe_dynamic_cast<VDBGrid>(get_input("vdbGrid"));
    auto attr = zsString2Std(get_input2_string("primAttr"));
    auto sampleby = zsString2Std(get_input2_string("sampleBy"));
    auto &pos = prim->attr<vec3f>(sampleby);
    auto type = zsString2Std(get_param_string("SampleType"));


    if (dynamic_cast<VDBFloatGrid *>(grid))
        prim->add_attr<float>(attr);
    else if (dynamic_cast<VDBFloat3Grid *>(grid))
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
        sampleVDBAttribute(pos, vel, grid); 
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
                                        {gParamType_VDBGrid,"vdbGrid", "", zeno::Socket_ReadOnly},
                                        {gParamType_String, "sampleBy","pos"},
                                        {gParamType_String, "primAttr", "sdf"}},
                                     {{gParamType_Primitive, "prim"}},
                                     {{"enum Clamp Periodic", "SampleType", "Clamp"}},
                                     {"openvdb"},
                                 });

static void primSampleVDB(
        PrimitiveObject* prim,
        const std::string &srcChannel,
        const std::string &dstChannel,
        VDBGrid* grid,
        float remapMin,
        float remapMax
) {
    auto &pos = prim->attr<vec3f>(srcChannel);
    if (dynamic_cast<VDBFloatGrid *>(grid)) {
        prim->add_attr<float>(dstChannel);
    }
    else if (dynamic_cast<VDBFloat3Grid *>(grid)) {
        prim->add_attr<vec3f>(dstChannel);
    }
    else {
        throw std::runtime_error("unknown vdb grid type");
    }
    prim->attr_visit(dstChannel, [&] (auto &vel) {
        if constexpr (is_vdb_to_prim_convertible<std::decay_t<decltype(vel)>>::value)
            sampleVDBAttribute2(pos, vel, grid, remapMin, remapMax);
    });
}

struct PrimSample3D : zeno::INode {
    virtual void apply() override {
        auto prim = clone_input_PrimitiveObject("prim");
        auto grid = safe_uniqueptr_cast<VDBGrid>(clone_input("vdbGrid"));
        auto dstChannel = zsString2Std(get_input2_string("dstChannel"));
        auto srcChannel = zsString2Std(get_input2_string("srcChannel"));
        auto remapMin = get_input2_float("remapMin");
        auto remapMax = get_input2_float("remapMax");

        primSampleVDB(prim.get(), srcChannel, dstChannel, grid.get(), remapMin, remapMax);
        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample3D, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_VDBGrid,"vdbGrid", "", zeno::Socket_ReadOnly},
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
        auto prim = clone_input_PrimitiveObject("prim");
        auto srcChannel = get_input2_string("srcChannel");
        auto dstChannel = get_input2_string("dstChannel");
        auto remapMin = get_input2_float("remapMin");
        auto remapMax = get_input2_float("remapMax");
        auto wrap = get_input2_string("wrap");
        auto borderColor = get_input2_vec3f("borderColor");
        if (has_input("sampledObject") && 
            get_input_PrimitiveObject("sampledObject")->userData()->has("isImage")) {
            auto image = clone_input_PrimitiveObject("sampledObject");
            primSampleTexture(prim.get(), srcChannel, "vertex", dstChannel, image.get(), wrap, borderColor, remapMin, remapMax);
        }
        else if (has_input("sampledObject")) {
            auto heatmap = safe_uniqueptr_cast<HeatmapObject>(clone_input("sampledObject"));
            primSampleHeatmap(prim.get(), srcChannel, dstChannel, heatmap.get(), remapMin, remapMax);
        }
        else if (has_input("sampledObject")) {
            auto grid = safe_dynamic_cast<VDBGrid>(get_input("vdbGrid"));
            primSampleVDB(prim.get(), zsString2Std(srcChannel), zsString2Std(dstChannel), grid, remapMin, remapMax);
        } else {
            throw zeno::Exception("unknown input type of sampledObject");
        }

        set_output("outPrim", std::move(prim));
    }
};
ZENDEFNODE(PrimSample, {
    {
        {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
        {gParamType_Primitive, "sampledObject", "", zeno::Socket_ReadOnly},
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
