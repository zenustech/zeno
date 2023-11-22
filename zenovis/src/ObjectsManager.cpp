#include <zenovis/ObjectsManager.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>
#include <zenovis/bate/IGraphic.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/log.h>
#include <zeno/types/MaterialObject.h>
#include <zeno/types/CameraObject.h>

std::set<std::string> lightCameraNodes({
    "CameraEval", "CameraNode", "CihouMayaCameraFov", "ExtractCameraData", "GetAlembicCamera","MakeCamera",
    "LightNode", "BindLight", "ProceduralSky", "HDRSky",
    });
std::string matlNode = "ShaderFinalize";

namespace zenovis {

ObjectsManager::ObjectsManager() = default;
ObjectsManager::~ObjectsManager() = default;

bool ObjectsManager::load_objects(std::map<std::string, std::shared_ptr<zeno::IObject>> const &objs) {
    bool inserted = false;
    auto ins = objects.insertPass();

    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            ins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }

    if (inserted || objs.size() < lastToViewNodesType.size())
        determineRenderType(objs);
    if (renderType != UNDEFINED) {
        lightObjects.clear();
        for (auto const& [key, obj] : objs) {
            if (auto prim_in = dynamic_cast<zeno::PrimitiveObject *>(obj.get())) {
                auto isRealTimeObject = prim_in->userData().get2<int>("isRealTimeObject", 0);
                if(isRealTimeObject){
                    //printf("loading light object %s\n", key.c_str());
                    lightObjects[key] = obj;
                }
            }
        }
    }

    return inserted;
}

void ObjectsManager::clear_objects() {
    objects.clear();
}

std::optional<zeno::IObject* > ObjectsManager::get(std::string nid) {
    for (auto &[key, ptr]: this->pairs()) {
        if (key != nid) {
            continue;
        }
        return ptr;
    }

    return std::nullopt;
}

void ObjectsManager::determineRenderType(std::map<std::string, std::shared_ptr<zeno::IObject>> const& objs)
{
    std::vector<size_t> count(3, 0);
    for (auto it = lastToViewNodesType.begin(); it != lastToViewNodesType.end();)
    {
        if (objs.find(it->first) == objs.end())
        {
            count[it->second]++;
            lastToViewNodesType.erase(it++);
        }
        else {
            it++;
        }
    }
    for (auto& [key, obj]: objs)
    {
        if (lastToViewNodesType.find(key) == lastToViewNodesType.end())
        {
            std::string nodeName = key.substr(key.find("-") + 1, key.find(":") - key.find("-") - 1);
            if (lightCameraNodes.count(nodeName) || obj->userData().get2<int>("isL", 0) || std::dynamic_pointer_cast<zeno::CameraObject>(obj)) {
                lastToViewNodesType.insert({ key, 0 });
                count[0]++;
            }
            else if (matlNode == nodeName || std::dynamic_pointer_cast<zeno::MaterialObject>(obj)) {
                lastToViewNodesType.insert({ key, 1 });
                count[1]++;
            }
            else {
                lastToViewNodesType.insert({ key, 2 });
                count[2]++;
            }
        }
    }
    renderType = count[1] == 0 && count[2] == 0 ? UPDATE_LIGHT_CAMERA : count[0] == 0 && count[2] == 0 ? UPDATE_MATERIAL : UPDATE_ALL;
}

}
