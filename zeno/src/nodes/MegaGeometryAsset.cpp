#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/MegaGeometryAsset.h>
#include <zeno/extra/assetDir.h>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <tinygltf/json.hpp>

namespace zeno {
struct MegaGeometryAsset : INode {
    void apply() override {
        const auto source=get_input2<std::string>("source",get_input2<std::string>("manifest",""));
        const auto jobs=get_input2<int>("bakeJobs",2),bakeRevision=get_input2<int>("bakeRevision",0);
        if(jobs<1 || jobs>64 || bakeRevision<0)throw std::runtime_error("Invalid Mega Geometry bake job count or revision");
        auto baker=std::filesystem::u8path(getExecutableDir())/"zeno_cluster_bake";
#ifdef _WIN32
        baker+=".exe";
#endif
        const auto path=prepareMegaGeometryAsset(std::filesystem::u8path(source),
            std::filesystem::u8path(get_input2<std::string>("cacheDirectory","")),baker,unsigned(jobs),unsigned(bakeRevision));
        std::ifstream stream(path,std::ios::binary);
        if(!stream) throw std::runtime_error("Cannot open Mega Geometry asset manifest: "+path.u8string());
        nlohmann::json meta;stream>>meta;
        if(meta.value("schema","")!="zeno.rtxmg.asset" || meta.value("version",0)!=1 || !meta.value("gray",false))
            throw std::runtime_error("MegaGeometryAsset requires a version-1 cluster bake");
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
        const bool materials=get_input2<bool>("loadMaterials",false);
        const int textureMiB=get_input2<int>("textureBudgetMiB",2048),hostMiB=get_input2<int>("textureHostMiB",256);
        if(materials && (textureMiB<64||hostMiB<32))throw std::runtime_error("CLAS material streaming needs at least 64 MiB texture and 32 MiB host budgets");
        ud.set2("clusterMaterials",materials);ud.set2("clusterTextureMiB",textureMiB);ud.set2("clusterTextureHostMiB",hostMiB);
        set_output("asset",std::move(ref));
    }
};
ZENDEFNODE(MegaGeometryAsset, {
    {{"readpath","source",""},{"string","objectName","MegaGeometry"},
     {"writepath","cacheDirectory",""},{"int","bakeJobs","2"},{"int","bakeRevision","0"},
     {"int","geometryBudgetMiB","2048"},{"int","clasBudgetMiB","2048"},
     {"float","lodPixelError","2"},{"int","loadsPerFrame","32"},
     {"bool","loadMaterials","1"},{"int","textureBudgetMiB","2048"},{"int","textureHostMiB","256"}},
    {"asset"},{},{"read"},
});
// Compatibility for external saved graphs. Authored scenes use the new node.
struct LoadPrebakedAsset : MegaGeometryAsset {};
ZENDEFNODE(LoadPrebakedAsset, {
    {{"readpath","manifest",""},{"string","objectName","ZorahGray"},
     {"int","geometryBudgetMiB","2048"},{"int","clasBudgetMiB","2048"},
     {"float","lodPixelError","2"},{"int","loadsPerFrame","32"},
     {"bool","loadMaterials","0"},{"int","textureBudgetMiB","2048"},{"int","textureHostMiB","256"}},
    {"asset"},{},{"read"},
});
}
