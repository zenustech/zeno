#pragma once

#include <map>
#include <vector>
#include <zeno/types/UserData.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/Scene.h>

namespace zenovis {

struct GraphicsManager {
    Scene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::unique_ptr<IGraphic>>>> graphics;
    zeno::PolymorphicMap<std::map<std::string, std::unique_ptr<IGraphic>>> realtime_graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool load_realtime_object(const std::string &key, std::shared_ptr<zeno::IObject> const &obj) {
        int interactive;
        if (obj->userData().has("interactive"))
            interactive = obj->userData().getLiterial<int>("interactive");
        else return false;
        if (interactive) {
            zeno::log_debug("load_realtime_object: loading realtime graphics [{}]", key);
            // printf("reload %s\n", key.c_str());
            auto ig = makeGraphic(scene, obj.get());
            zeno::log_debug("load_realtime_object: loaded realtime graphics to {}", ig.get());
            ig->nameid = key;
            ig->objholder = obj;
            realtime_graphics.try_emplace(key, std::move(ig));
            return true;
        }
        return false;
    }

    bool add_object(zeno::zany obj) {
        if (auto spList = std::dynamic_pointer_cast<zeno::ListObject>(obj)) {
            return add_listobj(spList);
        }
        if (auto spDict = std::dynamic_pointer_cast<zeno::DictObject>(obj)) {
            return add_dictobj(spDict);
        }

        const std::string& key = obj->key();
        if (!obj || key.empty())
            return false;

        auto& wtf = graphics.m_curr.m_curr;
        auto it = wtf.find(key);
        if (it == wtf.end()) {
            zeno::log_debug("load_object: loading graphics [{}]", key);
            auto ig = makeGraphic(scene, obj.get());
            if (!ig)
                return false;
            zeno::log_debug("load_object: loaded graphics to {}", ig.get());
            ig->nameid = key;
            ig->objholder = obj;
            graphics.m_curr.m_curr.insert(std::make_pair(key, std::move(ig)));
        }
        else {
            auto ig = makeGraphic(scene, obj.get());
            if (!ig)
                return false;
            ig->nameid = key;
            ig->objholder = obj;
            it->second = std::move(ig);
        }
        return true;
    }

    bool remove_object(zeno::zany spObj) {
        if (auto spList = std::dynamic_pointer_cast<zeno::ListObject>(spObj)) {
            return remove_listobj(spList);
        }
        if (auto spDict = std::dynamic_pointer_cast<zeno::DictObject>(spObj)) {
            return remove_dictobj(spDict);
        }

        const std::string& key = spObj->key();
        auto& graphics_ = graphics.m_curr.m_curr;
        auto iter = graphics_.find(key);
        if (iter == graphics_.end())
            return false;

        graphics_.erase(key);
        return true;
    }

    bool add_listobj(std::shared_ptr<zeno::ListObject> spList) {
        for (auto obj : spList->get()) {
            if (auto listobj = std::dynamic_pointer_cast<zeno::ListObject>(obj)) {
                bool ret = add_listobj(listobj);
                if (!ret)
                    return ret;
            }
            else {
                bool ret = add_object(obj);
                if (!ret)
                    return ret;
            }
        }
        return true;
    }

    bool add_dictobj(std::shared_ptr<zeno::DictObject> spDict) {
        for (auto& [key, spObject] : spDict->get()) {
            if (auto dictobj = std::dynamic_pointer_cast<zeno::DictObject>(spObject)) {
                bool ret = add_dictobj(dictobj);
                if (!ret)
                    return ret;
            }
            else {
                bool ret = add_object(spObject);
                if (!ret)
                    return ret;
            }
        }
        return true;
    }

    bool remove_listobj(std::shared_ptr<zeno::ListObject> spList) {
        for (auto obj : spList->get()) {
            if (auto listobj = std::dynamic_pointer_cast<zeno::ListObject>(obj)) {
                bool ret = remove_listobj(listobj);
                if (!ret)
                    return ret;
            }
            else {
                bool ret = remove_object(obj);
                if (!ret)
                    return ret;
            }
        }
        return true;
    }

    bool remove_dictobj(std::shared_ptr<zeno::DictObject> spDict) {
        for (auto& [key, spObject] : spDict->get()) {
            if (auto dictobj = std::dynamic_pointer_cast<zeno::DictObject>(spObject)) {
                bool ret = remove_dictobj(dictobj);
                if (!ret)
                    return ret;
            }
            else {
                bool ret = remove_object(spObject);
                if (!ret)
                    return ret;
            }
        }
        return true;
    }

    void load_objects2(const zeno::RenderObjsInfo& objs) {
        for (auto [key, spObj] : objs.remObjs) {    //if obj both in remObjs and in newObjs, need remove first?
            remove_object(spObj);
        }
        for (auto [key, spObj] : objs.newObjs) {
            add_object(spObj);
        }
        for (auto [key, spObj] : objs.modifyObjs) {
            bool isListDict = false;
            if (auto spList = std::dynamic_pointer_cast<zeno::ListObject>(spObj)) {
                isListDict = true;
            } else if (auto spDict = std::dynamic_pointer_cast<zeno::DictObject>(spObj)) {
                isListDict = true;
            }
            if (isListDict) {
                auto& wtf = graphics.m_curr.m_curr;
                for (auto it = wtf.begin(); it != wtf.end(); ) {
                    if (it->first.find(key) != std::string::npos)
                        it = wtf.erase(it);
                    else
                        ++it;
                }
            }
            add_object(spObj);
        }
    }

    //deprecated
    bool load_objects(std::vector<std::pair<std::string, std::shared_ptr<zeno::IObject>>> const &objs) {
        auto ins = graphics.insertPass();
        realtime_graphics.clear();
        for (auto const &[key, obj] : objs) {
            if (load_realtime_object(key, obj)) continue;
            if (ins.may_emplace(key)) {
                zeno::log_debug("load_object: loading graphics [{}]", key);
                auto ig = makeGraphic(scene, obj.get());
                zeno::log_debug("load_object: loaded graphics to {}", ig.get());
                ig->nameid = key;
                ig->objholder = obj;
                ins.try_emplace(key, std::move(ig));
            }
        }
        return ins.has_changed();
    }

    void draw() {
        for (auto const &[key, gra] : graphics.pairs<IGraphicDraw>()) {
            // if (realtime_graphics.find(key) == realtime_graphics.end())
            gra->draw();
        }
        //for (auto const &[key, gra] : realtime_graphics.pairs<IGraphicDraw>()) {
        //    gra->draw();
        //}
        // printf("graphics count: %d\n", graphics.size());
        // printf("realtime graphics count: %d\n", realtime_graphics.size());
    }
};

} // namespace zenovis
