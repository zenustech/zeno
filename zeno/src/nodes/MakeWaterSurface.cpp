#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <tinygltf/json.hpp>
#include <cmath>

namespace zeno {
struct MakeWaterSurface : INode {
    void apply() override {
        const auto center=get_input2<vec3f>("center");
        const auto size=get_input2<vec2f>("size");
        const auto wind=get_input2<vec2f>("windDirection");
        nlohmann::json settings{{"center",{center[0],center[1],center[2]}},{"size",{size[0],size[1]}},
            {"wind",{wind[0],wind[1]}}};
        for(const char* name:{"time","amplitude","period","windSpeed","depth","edgePixels","boundaryWidth"}) {
            const auto value=get_input2<float>(name);
            if(!std::isfinite(value))throw std::runtime_error(std::string("Nonfinite water parameter: ")+name);
            settings[name]=value;
        }
        const float loop=get_input2<float>("loopDuration",0.f);
        if(!std::isfinite(loop)||loop<0.f)throw std::runtime_error("Water loop duration must be finite and nonnegative");
        settings["loopDuration"]=loop;
        // Cache only a small descriptor per graph frame. The renderer retains
        // the spectrum, FFT plan and GPU topology across descriptor changes.
        if(get_input2<bool>("animate",false)) {
            const auto* state=getGlobalState();
            const double seconds=double(state->frameid)*state->frame_time+state->frame_time_elapsed;
            settings["time"]=settings["time"].get<double>()+seconds;
        }
        for(const char* name:{"resolution","seed","maxClusters"}) {
            const auto value=get_input2<int>(name);
            if(value<0)throw std::runtime_error(std::string("Negative water parameter: ")+name);
            settings[name]=value;
        }
        if(size[0]<=0||size[1]<=0)throw std::runtime_error("Water dimensions must be positive");
        for(float value:center)if(!std::isfinite(value))throw std::runtime_error("Water center must be finite");
        for(float value:size)if(!std::isfinite(value))throw std::runtime_error("Water size must be finite");
        for(float value:wind)if(!std::isfinite(value))throw std::runtime_error("Water wind must be finite");
        settings["ellipse"]=get_input2<bool>("ellipse")?1:0;
        auto object=std::make_shared<PrimitiveObject>();
        object->userData().set2("ResourceType",std::string("WaterSurface"));
        object->userData().set2("ObjectName",get_input2<std::string>("objectName"));
        object->userData().set2("waterSettings",settings.dump());
        set_output("surface",std::move(object));
    }
};
ZENDEFNODE(MakeWaterSurface,{
    {{"vec3f","center","0,0,0"},{"vec2f","size","10,10"},{"bool","ellipse","0"},
     {"bool","animate","0"},{"float","time","0"},{"float","loopDuration","0"},
     {"float","amplitude","0.008"},{"float","period","12"},
     {"vec2f","windDirection","1,0"},{"float","windSpeed","3"},{"float","depth","1"},
     {"int","resolution","256"},{"int","seed","42"},{"float","edgePixels","2"},{"int","maxClusters","8192"},
     {"float","boundaryWidth","0.1"},{"string","objectName","Water"}},
    {"surface"},{},{"primitive"},
});
}
