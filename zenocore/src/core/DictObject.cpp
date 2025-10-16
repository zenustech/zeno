#include <zeno/types/DictObject.h>
#include <zeno/types/UserData.h>


namespace zeno
{
    DictObject::DictObject() {
    }

    DictObject::DictObject(const DictObject& dictObj) {
        m_key = dictObj.m_key;
        for (auto& [str, obj] : dictObj.lut) {
            auto cloneobj = obj->clone();
            cloneobj->update_key(obj->key());
            lut.insert({ str, std::move(cloneobj) });
        }
        m_modify = dictObj.m_modify;
        m_new_added = dictObj.m_new_added;
        m_new_removed = dictObj.m_new_removed;
    }

    DictObject& DictObject::operator=(const DictObject& dictObj) {
        m_key = dictObj.m_key;
        for (auto& [str, obj] : dictObj.lut) {
            auto cloneobj = obj->clone();
            cloneobj->update_key(obj->key());
            lut.insert({ str, std::move(cloneobj) });
        }
        m_modify = dictObj.m_modify;
        m_new_added = dictObj.m_new_added;
        m_new_removed = dictObj.m_new_removed;
        return *this;
    }

    DictObject::~DictObject() {
        //由于IObject没有虚析构（保证abi兼容），所以一般这里不会走到这，除非是用户强行构造实例，这种情况还是要手动清理
        clear_children();
    }

    void DictObject::clear_children() {
        //由于目前object的内存管理方式还是基于shared_ptr，故不能直接手动Delete.
        /*
        for (auto& [key, pObj] : lut) {
            pObj->Delete();
        }
        */
        lut.clear();
    }

    void DictObject::Delete() {
    }


    std::unique_ptr<DictObject> create_DictObject() {
        return std::make_unique<DictObject>();
    }
}