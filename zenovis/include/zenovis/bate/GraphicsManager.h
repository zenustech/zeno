#pragma once

#include <map>
#include <vector>
#include <zeno/types/UserData.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/extra/SceneAssembler.h>
#include <zeno/utils/MapStablizer.h>
#include <zeno/utils/PolymorphicMap.h>
#include <zeno/utils/log.h>
#include <zenovis/bate/IGraphic.h>
#include <zenovis/Scene.h>
#include <zeno/core/Graph.h>

namespace zenovis {

struct GraphicsManager {
    Scene *scene;

    zeno::MapStablizer<zeno::PolymorphicMap<std::map<
        std::string, std::unique_ptr<IGraphic>>>> graphics;
    zeno::PolymorphicMap<std::map<std::string, std::unique_ptr<IGraphic>>> realtime_graphics;

    explicit GraphicsManager(Scene *scene) : scene(scene) {
    }

    bool add_object(zeno::IObject* obj) {
        if (auto spList = dynamic_cast<zeno::ListObject*>(obj)) {
            for (auto elemObj : spList->m_impl->get()) {
                bool ret = add_object(elemObj);
                //assert(ret);  //有一些是JsonObject，不应该进来
            }
            return true;
        }
        if (auto spDict = dynamic_cast<zeno::DictObject*>(obj)) {
            for (auto& [key, spObject] : spDict->get()) {
                bool ret = add_object(spObject);
                assert(ret);
            }
            return true;
        }

        const std::string& key = zsString2Std(obj->key());
        if (!obj || key.empty())
            return false;

        auto& wtf = graphics.m_curr.m_curr;
        auto it = wtf.find(key);
        if (it == wtf.end()) {
            zeno::log_debug("load_object: loading graphics [{}]", key);
            auto ig = makeGraphic(scene, obj);
            if (!ig)
                return false;
            zeno::log_debug("load_object: loaded graphics to {}", ig.get());
            ig->nameid = key;
            graphics.m_curr.m_curr.insert(std::make_pair(key, std::move(ig)));
        }
        else {
            auto ig = makeGraphic(scene, obj);
            if (!ig)
                return false;
            ig->nameid = key;
            it->second = std::move(ig);
        }
        return true;
    }

    bool remove_object_bykey(const std::string& key) {
        auto& graphics_ = graphics.m_curr.m_curr;
        auto iter = graphics_.find(key);
        if (iter == graphics_.end())
            return false;

        graphics_.erase(key);
        return true;
    }

    bool remove_object(zeno::IObject* spObj) {
        if (auto spList = dynamic_cast<zeno::ListObject*>(spObj)) {
            for (auto obj : spList->m_impl->get()) {
                bool ret = remove_object(obj);
                assert(ret);
            }
            return true;
        }
        if (auto spDict = dynamic_cast<zeno::DictObject*>(spObj)) {
            for (auto& [key, spObject] : spDict->get()) {
                bool ret = remove_object(spObject);
                assert(ret);
            }
            return true;
        }
        const std::string& key = zsString2Std(spObj->key());
        auto& graphics_ = graphics.m_curr.m_curr;
        auto iter = graphics_.find(key);
        if (iter == graphics_.end())
            return false;

        graphics_.erase(key);
        return true;
    }

    bool process_listobj(zeno::ListObject* spList, bool bProcessAll = false) {
        //由于现在不再统计容器内移除的对象，因此需要在这里收集旧的信息，然后和新的作对比
        auto rootkey = zsString2Std(spList->key());
        //收集list相关的key
        auto& graphics_ = graphics.m_curr.m_curr;

        for (auto spObject : spList->m_impl->get()) {
            std::string const& key = zsString2Std(spObject->key());
            if (bProcessAll ||
                (spList->m_impl->m_new_added.find(key) != spList->m_impl->m_new_added.end() ||
                 spList->m_impl->m_modify.find(key) != spList->m_impl->m_modify.end()))
            {
                if (auto _spList = dynamic_cast<zeno::ListObject*>(spObject)) {
                    process_listobj(_spList, bProcessAll);
                }
                else if (auto _spDict = dynamic_cast<zeno::DictObject*>(spObject)) {
                    process_dictobj(_spDict);
                }
                else {
                    add_object(spObject);
                }
            }
        }
        for (auto& key : spList->m_impl->m_new_removed) {
            graphics_.erase(key);
        }
        return true;
    }

    bool process_dictobj(zeno::DictObject* spDict) {
        for (auto& [key, spObject] : spDict->get()) {
            assert(spObject);
            std::string const& skey = zsString2Std(spObject->key());
            if (spDict->m_new_added.find(skey) != spDict->m_new_added.end() ||
                spDict->m_modify.find(skey) != spDict->m_modify.end()) {
                bool ret = false;
                if (auto _spList = dynamic_cast<zeno::ListObject*>(spObject)) {
                    ret = process_listobj(_spList);
                }
                else if (auto _spDict = dynamic_cast<zeno::DictObject*>(spObject)) {
                    ret = process_dictobj(_spDict);
                }
                else {
                    ret = add_object(spObject);
                }
                assert(ret);
            }
        }
        for (auto& key : spDict->m_new_removed) {
            auto& graphics_ = graphics.m_curr.m_curr;
            auto iter = graphics_.find(key);
            if (iter == graphics_.end())
                continue;
            graphics_.erase(key);
        }
        return true;
    }

    void reload(const zeno::render_reload_info& info) {
        auto& sess = zeno::getSession();
        if (zeno::Reload_SwitchGraph == info.policy) {
            //由于对象和节点是一一对应，故切换图层次结构必然导致所有对象被重绘
            graphics.clear();
            std::shared_ptr<zeno::Graph> spGraph = sess.getGraphByPath(info.current_ui_graph);
            if (spGraph) {
                if (spGraph->isAssets()) {
                    //资产图不能view，因为没有实例化，不属于运行图的范畴
                    return;
                }
                const auto& viewnodes = spGraph->get_viewnodes();
                //其实是否可以在外面提前准备好对象列表？
                for (auto viewnode : viewnodes) {
                    auto spNode = spGraph->getNode(viewnode);
                    auto spObject = spNode->get_default_output_object();
                    if (spObject) {
                        add_object(spObject);
                    }
                    else {

                    }
                }
            }
        }
        else if (zeno::Reload_ToggleView == info.policy) {
            if (info.objs.size() != 1) {
                //TODO: 谁说view只能一个的？？
                return;
            }
            const auto& update = info.objs[0];
            if (update.reason == zeno::Update_Remove) {
                for (const std::string& remkey : update.remove_objs) {
                    remove_object_bykey(remkey);
                }
            }
            else {
                if (update.spObject) {
                    auto spObject = update.spObject.get();
                    if (update.reason == zeno::Update_View) {
                        add_object(spObject);
                    }
                }
                else {
                    //可能还没计算
                }
            }
        }
        else if (zeno::Reload_Calculation == info.policy) {
            for (const zeno::render_update_info& update : info.objs) {
                auto spObject = update.spObject.get();
                if (spObject) {
                    //可能是对象没有通过子图的Suboutput连出来

                    if (auto sceneObj = dynamic_cast<zeno::SceneObject*>(spObject)) {
                        auto _spList = sceneObj->to_structure();
                        _spList->update_key(sceneObj->key());
                        process_listobj(_spList.get(), true);
                    }
                    else if (auto _spList = dynamic_cast<zeno::ListObject*>(spObject)) {

                        {//可能有和listobj同名但不是list类型的对象存在，需先清除
                            auto& graphics_ = graphics.m_curr.m_curr;
                            std::string listkey = zsString2Std(_spList->key());
                            const auto& it = listkey.find('\\');
                            const std::string& key = it == std::string::npos ? listkey : listkey.substr(0, it);
                            for (auto it = graphics_.begin(); it != graphics_.end(); ) {
                                if (it->first == key)
                                    it = graphics_.erase(it);
                                else
                                    ++it;
                            }
                        }

                        process_listobj(_spList);
                    }
                    else if (auto _spDict = dynamic_cast<zeno::DictObject*>(spObject)) {
                        {//可能有和dictobj同名但不是dict类型的对象存在，需先清除
                            auto& graphics_ = graphics.m_curr.m_curr;
                            std::string dictkey = zsString2Std(_spDict->key());
                            const auto& it = dictkey.find('\\');
                            const std::string& key = it == std::string::npos ? dictkey : dictkey.substr(0, it);
                            for (auto it = graphics_.begin(); it != graphics_.end(); ) {
                                if (it->first == key)
                                    it = graphics_.erase(it);
                                else
                                    ++it;
                            }
                        }

                        process_dictobj(_spDict);
                    }
                    else {
                        {//可能有和obj同名但是list类型或dict类型的对象存在，需先清除
                            auto& graphics_ = graphics.m_curr.m_curr;
                            std::string objkey = zsString2Std(spObject->key());
                            for (auto it = graphics_.begin(); it != graphics_.end(); ) {
                                if (it->first.find(objkey + '\\') != std::string::npos) {
                                    it = graphics_.erase(it);
                                }
                                else
                                    ++it;
                            }
                        }

                        add_object(spObject);
                    }
                }
            }
        }
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
