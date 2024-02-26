#include <zeno/core/ObjectManager.h>


namespace zeno {

    ZENO_API std::recursive_mutex g_objsMutex;

    ObjectManager::ObjectManager()
    {
    }

    ObjectManager::~ObjectManager()
    {
    }

    ZENO_API void ObjectManager::addObject(const std::string& id, std::shared_ptr<IObject> obj, std::shared_ptr<INode> view_node, bool bView)
    {
        std::lock_guard lck(g_objsMutex);
        auto it = m_objects.find(id);
        if (it == m_objects.end()) {
            _ObjInfo info;
            info.obj = obj;
            info.view_node = view_node;
            m_objects.insert(std::make_pair(id, info));
        }
        else {
            it->second.obj = obj;
            it->second.view_node = view_node;
        }
        CALLBACK_NOTIFY(addObject, obj, bView)
    }

    ZENO_API void ObjectManager::removeObject(const std::string& id)
    {
        std::lock_guard lck(g_objsMutex);
        if (m_objects.find(id) != m_objects.end()) {
            m_objects.erase(id);
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

    void ObjectManager::clear()
    {
        //todo
    }

}