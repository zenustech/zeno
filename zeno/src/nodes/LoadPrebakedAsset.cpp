#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <tinygltf/json.hpp>

namespace zeno {
struct LoadPrebakedAsset : INode {
    void apply() override {
        const auto path=std::filesystem::absolute(std::filesystem::u8path(get_input2<std::string>("manifest"))).lexically_normal();
        std::ifstream stream(path,std::ios::binary);
        if(!stream) throw std::runtime_error("Cannot open prebaked asset manifest: "+path.u8string());
        nlohmann::json meta;stream>>meta;
        if(meta.value("schema","")!="zeno.rtxmg.asset" || meta.value("version",0)!=1 || !meta.value("gray",false))
            throw std::runtime_error("LoadPrebakedAsset requires a version-1 gray cluster bake, not a source-selection manifest");
        const int geometry=get_input2<int>("geometryBudgetMiB"),clas=get_input2<int>("clasBudgetMiB"),loads=get_input2<int>("loadsPerFrame");
        const float error=get_input2<float>("lodPixelError");
        if(geometry<=0||clas<=0||loads<=0||!std::isfinite(error)||error<=0)
            throw std::runtime_error("Prebaked asset requires positive streaming budgets, loads and LOD error");
        auto ref=std::make_shared<PrimitiveObject>(); // Descriptor only: no geometry arrays.
        auto& ud=ref->userData();
        ud.set2("ResourceType",std::string("ClusterAsset"));
        ud.set2("ObjectName",get_input2<std::string>("objectName"));
        ud.set2("clusterManifest",path.u8string());
        uint64_t revision=14695981039346656037ull;
        for(unsigned char byte:meta.dump()){revision^=byte;revision*=1099511628211ull;}
        ud.set2("clusterRevision",std::to_string(revision));
        ud.set2("clusterGeometryMiB",geometry);ud.set2("clusterClasMiB",clas);
        ud.set2("clusterLodError",error);ud.set2("clusterLoadsPerFrame",loads);
        set_output("asset",std::move(ref));
    }
};
ZENDEFNODE(LoadPrebakedAsset, {
    {{"readpath","manifest",""},{"string","objectName","ZorahGray"},
     {"int","geometryBudgetMiB","2048"},{"int","clasBudgetMiB","2048"},
     {"float","lodPixelError","2"},{"int","loadsPerFrame","32"}},
    {"asset"},{},{"read"},
});
}
