#include <zeno/core/ObjectManager.h>


namespace zeno {

    ZENO_API std::recursive_mutex g_objsMutex;

    ObjectManager::ObjectManager()
    {
    }

    ObjectManager::~ObjectManager()
    {
    }

    ZENO_API void ObjectManager::commit()
    {
        std::lock_guard lck(g_objsMutex);
        m_commitRender = m_collecting;
        m_collecting.clear();
        auto& pGlobalStatue = zeno::getSession().globalState;

        while (pGlobalStatue->getCalcObjStatus() == zeno::Loading) {
            //waiting.
        }

        pGlobalStatue->setCalcObjStatus(zeno::Finished);
    }

    ZENO_API void ObjectManager::revert()
    {

    }

    ZENO_API void ObjectManager::collectingObject(const std::string& id, std::shared_ptr<IObject> obj, std::shared_ptr<INode> view_node, bool bView)
    {
        std::lock_guard lck(g_objsMutex);

        zeno::getSession().globalState->setCalcObjStatus(zeno::Collecting);

        auto it = m_collecting.find(id);
        auto path = view_node->get_path();
        if (it == m_collecting.end()) {
            _ObjInfo info;
            info.obj = obj;
            info.attach_nodes.insert(path);
            m_collecting.insert(std::make_pair(id, info));
        }
        else {
            it->second.obj = obj;
            it->second.attach_nodes.insert(path);
        }
        if (bView) {
            m_viewObjs.insert(id);
            m_lastViewObjs.erase(id);   //上一次运行有view，这一次也有view
        }
        else {
        }
        CALLBACK_NOTIFY(collectingObject, obj, bView)
    }

    ZENO_API void ObjectManager::removeObject(const std::string& id)
    {
        std::lock_guard lck(g_objsMutex);
        if (m_collecting.find(id) != m_collecting.end()) {
            m_collecting.erase(id);
            CALLBACK_NOTIFY(removeObject, id)
        }
    }

    ZENO_API void ObjectManager::notifyTransfer(std::shared_ptr<IObject> obj)
    {
        std::lock_guard lck(g_objsMutex);
        CALLBACK_NOTIFY(notifyTransfer, obj)
    }

    ZENO_API void ObjectManager::viewObject(std::shared_ptr<IObject> obj, bool bView)
    {
        std::lock_guard lck(g_objsMutex);
        CALLBACK_NOTIFY(viewObject, obj, bView)
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
        auto it = m_collecting.find(id);
        if (it != m_collecting.end())
        {
            return it->second.attach_nodes;
        }
        return std::set<ObjPath>();
    }

    ZENO_API void ObjectManager::beforeRun()
    {
        m_lastViewObjs = m_viewObjs;
        m_viewObjs.clear();
    }

    ZENO_API void ObjectManager::afterRun()
    {
        //剩下来的都是上一次view，而这一次没有view的。
        for (auto objkey : m_lastViewObjs)
        {
            removeObject(objkey);
        }
        m_lastViewObjs.clear();
    }

    void ObjectManager::clear()
    {
        //todo
    }

}