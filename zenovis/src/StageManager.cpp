#include <zenovis/StageManager.h>

#include <iostream>

PXR_NAMESPACE_USING_DIRECTIVE

namespace zenovis {

StageManager::StageManager(){
    zeno::log_info("USD: StageManager Constructed");

    UsdStageRefPtr visStage = UsdStage::CreateInMemory();
};
StageManager::~StageManager(){
    zeno::log_info("USD: StageManager Destroyed");
};

bool zenovis::StageManager::load_objects(const std::map<std::string, std::shared_ptr<zeno::IObject>> &objs) {
    zeno::log_info("USD: StageManager load_objects");
    auto ins = zenoObjects.insertPass();
    bool inserted = false;
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {
            ins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }
    return inserted;
}

}