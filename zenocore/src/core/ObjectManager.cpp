#include <zeno/core/ObjectManager.h>
#include <zeno/core/Graph.h>
#include <zeno/types/ListObject.h>


namespace zeno {

    ObjectManager::ObjectManager()
    {
    }

    ObjectManager::~ObjectManager()
    {
    }

    ZENO_API void ObjectManager::commit()
    {
    }

    ZENO_API void ObjectManager::revert()
    {
    }

    ZENO_API void ObjectManager::collectingObject(std::shared_ptr<IObject> obj, std::shared_ptr<INodeAny> anynode, bool bView)
    {
        std::lock_guard lck(m_mtx);

        const std::string& objId = obj->key();
        if (objId.empty())
            return;

        //zeno::getSession().globalState->setCalcObjStatus(zeno::Collecting);
        auto it = m_objects.find(objId);
        auto attachNode = AnyToINodePtr(anynode);
        auto path = attachNode->get_uuid_path();
        bool bExist = it != m_objects.end();
        if (!bExist) {
            _ObjInfo info;
            info.obj = obj;
            info.attach_nodes.insert(path);
            m_objects.insert(std::make_pair(objId, info));
        }
        else {
            it->second.obj = obj;
            it->second.attach_nodes.insert(path);
        }

        //仅处理view的对象，不view的可能没必要加进来。
        if (bView) {
            m_viewObjs.insert(objId);
            if (m_lastViewObjs.find(objId) != m_lastViewObjs.end()) {
                m_lastViewObjs.erase(objId);
                //上一次运行有view，这一次也有view，如果节点状态是运行态，说明是修改对象
                if (Node_Running == attachNode->get_run_status()) {
                    m_modify.insert(objId);
                }
            }
            else {
                //上一次没有view，这次有view，要么就是新增，要么就是重新打view
                m_newAdded.insert(objId);
            }
        }
    }

    ZENO_API void ObjectManager::removeObject(const std::string& id)
    {
        std::lock_guard lck(m_mtx);
        m_lastUnregisterObjs.insert(id); //先标记，下一次run的时候在去m_objects中移除
    }

    ZENO_API void ObjectManager::revertRemoveObject(const std::string& id)
    {
        std::lock_guard lck(m_mtx);
        m_lastUnregisterObjs.erase(id); //有一种情况是apply时仅对obj进行modify，此时需要将apply之前加入的待删除obj的id移除，无需下次运行时清除该obj
    }

    ZENO_API void ObjectManager::notifyTransfer(std::shared_ptr<IObject> obj)
    {
        //std::lock_guard lck(m_mtx);
        //CALLBACK_NOTIFY(notifyTransfer, obj)
    }

    ZENO_API void ObjectManager::viewObject(std::shared_ptr<IObject> obj, bool bView)
    {
        //std::lock_guard lck(m_mtx);
    }

    ZENO_API int ObjectManager::registerObjId(const std::string& objprefix)
    {
        if (m_objRegister.find(objprefix) == m_objRegister.end()) {
            m_objRegister.insert(std::make_pair(objprefix, 0));
            m_objRegister[objprefix]++;
            return 0;
        }
        else {
            int newObjId = m_objRegister[objprefix]++;
            return newObjId;
        }
    }

    ZENO_API std::set<ObjPath> ObjectManager::getAttachNodes(const std::string& id)
    {
        std::lock_guard lck(m_mtx);
        auto it = m_objects.find(id);
        if (it != m_objects.end())
        {
            return it->second.attach_nodes;
        }
        return std::set<ObjPath>();
    }

    ZENO_API void ObjectManager::beforeRun()
    {
        std::lock_guard lck(m_mtx);     //可能此时渲染端在load_objects
        m_lastViewObjs = m_viewObjs;
        m_viewObjs.clear();
        m_newAdded.clear();
        m_modify.clear();
        m_remove.clear();
    }

    ZENO_API void ObjectManager::afterRun()
    {
        std::lock_guard lck(m_mtx);
        //m_lastViewObjs剩下来的都是上一次view，而这一次没有view的。
        m_remove = m_lastViewObjs;
        m_lastViewObjs.clear();
        m_removing_objs.clear();
    }

    ZENO_API void ObjectManager::clearLastUnregisterObjs()
    {
        for (auto& key : m_lastUnregisterObjs)
            if (m_objects.find(key) != m_objects.end())
                m_objects.erase(key);
        m_lastUnregisterObjs.clear();
    }

    ZENO_API void ObjectManager::clear()
    {
        m_objects.clear();
        m_viewObjs.clear();
        m_lastViewObjs.clear();
        m_removing_objs.clear();
        m_newAdded.clear();
        m_remove.clear();
        m_modify.clear();
        m_frameData.clear();
    }

    ZENO_API void ObjectManager::clear_last_run()
    {
        std::lock_guard lck(m_mtx);
        m_newAdded.clear();
        m_modify.clear();
        m_remove.clear();
    }

    ZENO_API void ObjectManager::collect_removing_objs(const std::string& objkey)
    {
        m_removing_objs.insert(objkey);
    }

    ZENO_API void ObjectManager::remove_attach_node_by_removing_objs()
    {
        for (auto obj_key : m_removing_objs) {
            auto nodes = getAttachNodes(obj_key);
            for (auto node_path : nodes) {
                auto spNode = AnyToINodePtr(zeno::getSession().mainGraph->getNodeByUuidPath(node_path));
                if (spNode)
                    spNode->mark_dirty(true);
            }
            removeObject(obj_key);
        }
    }

    ZENO_API void ObjectManager::remove_rendering_obj(zany spObj)
    {
        std::string key = spObj->key();
        if (key.empty())
            return;
        m_remove.insert(key);
    }

    ZENO_API void ObjectManager::collect_modify_objs(const std::set<std::string>& newobjKeys, bool isView)
    {
        std::lock_guard lck(m_mtx);
        if (isView)
            m_modify.insert(newobjKeys.begin(), newobjKeys.end());;
    }

    ZENO_API void ObjectManager::remove_modify_objs(const std::set<std::string>& removeobjKeys)
    {
        std::lock_guard lck(m_mtx);
        m_modify.clear();
    }

    ZENO_API void ObjectManager::collect_modify_objs(const std::string& newobjKey, bool isView)
    {
        std::lock_guard lck(m_mtx);
        if (isView)
            m_modify.insert(newobjKey);;
    }

    ZENO_API void ObjectManager::getModifyObjsInfo(std::set<std::string>& modifyInteractiveObjs)
    {
        std::lock_guard lck(m_mtx);
        modifyInteractiveObjs = m_modify;
    }

    ZENO_API void ObjectManager::export_loading_objs(RenderObjsInfo& info)
    {
        std::lock_guard lck(m_mtx);
        for (auto objkey : m_newAdded) {
            auto it = m_objects.find(objkey);
            if (it != m_objects.end())
                info.newObjs.insert(std::make_pair(objkey, it->second.obj));
        }
        for (auto objkey : m_modify) {
            auto it = m_objects.find(objkey);
            if (it != m_objects.end())
                info.modifyObjs.insert(std::make_pair(objkey, it->second.obj));
        }
        for (auto objkey : m_remove) {
            auto it = m_objects.find(objkey);
            if (it != m_objects.end())
                info.remObjs.insert(std::make_pair(objkey, it->second.obj));
        }
    }

    ZENO_API void ObjectManager::export_all_view_objs(RenderObjsInfo& info)
    {
        std::lock_guard lck(m_mtx);
        for (auto& key : m_viewObjs) {
            auto& it = m_objects.find(key);
            if (it != m_objects.end())
                info.allObjects.emplace(std::move(std::pair(key, it->second.obj)));
        }
    }

    ZENO_API void ObjectManager::export_all_view_objs(std::map<std::string, std::shared_ptr<zeno::IObject>>& info)
    {
        std::lock_guard lck(m_mtx);
        for (auto& key : m_viewObjs) {
            auto& it = m_objects.find(key);
            if (it != m_objects.end())
                info.emplace(std::make_pair(key, it->second.obj));
        }
    }

    ZENO_API std::shared_ptr<zeno::IObject> ObjectManager::getObj(const std::string& name)
    {
        std::lock_guard lck(m_mtx);
        if (m_objects.find(name) != m_objects.end())
            return m_objects[name].obj;
        return nullptr;
    }

    ZENO_API ObjectNodeInfo ObjectManager::getObjectAndViewNode(const std::string& name)
    {
        ObjectNodeInfo info;

        std::lock_guard lck(m_mtx);
        auto iter = m_objects.find(name);
        if (iter == m_objects.end())
            return info;

        info.transformingObj = iter->second.obj;
        auto& mainG = getSession().mainGraph;
        for (auto nodepath : iter->second.attach_nodes) {
            auto spAnyNode = mainG->getNodeByUuidPath(nodepath);
            auto spNode = AnyToINodePtr(spAnyNode);
            if (spNode->is_view())
            {
                info.spViewNode = spAnyNode;
                break;
            }
        }
        return info;
    }

}