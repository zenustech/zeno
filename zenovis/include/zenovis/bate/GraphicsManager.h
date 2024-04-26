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

    bool add_object(std::shared_ptr<zeno::IObject> obj) {

        if (auto spList = std::dynamic_pointer_cast<zeno::ListObject>(obj)) {
            return add_listobj(spList);
        }

        if (!obj || obj->key().empty()) return false;

        auto& wtf = graphics.m_curr.m_curr;
        auto it = wtf.find(obj->key());
        if (it == wtf.end()) {
            zeno::log_debug("load_object: loading graphics [{}]", obj->key());
            auto ig = makeGraphic(scene, obj.get());
            if (!ig)
                return false;
            zeno::log_debug("load_object: loaded graphics to {}", ig.get());
            ig->nameid = obj->key();
            ig->objholder = obj;
            graphics.m_curr.m_curr.insert(std::make_pair(obj->key(), std::move(ig)));
        }
        else {
            auto ig = makeGraphic(scene, obj.get());
            if (!ig)
                return false;
            ig->nameid = obj->key();
            ig->objholder = obj;
            it->second = std::move(ig);
        }
        return true;
    }

    bool remove_object(std::string key) {
        auto& wtf = graphics.m_curr.m_curr;
        if (wtf.find(key) == wtf.end())
            return false;
        wtf.erase(key);
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

    void load_objects2(const zeno::RenderObjsInfo& objs) {
        std::map<std::string, std::shared_ptr<zeno::IObject>> listItems;
        std::set<std::string> removeListItems;                                              //本次运行list中要删除的元素

        //处理单个Object
        for (auto [key, spObj] : objs.newObjs) {
            add_object(spObj);
        }
        for (auto [key, spObj] : objs.modifyObjs) {
            add_object(spObj);
        }
        for (auto [key, spObj] : objs.remObjs) {
            remove_object(key);
        }
    }

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
