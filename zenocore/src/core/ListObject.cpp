#include <zeno/utils/helper.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/ListObject_impl.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/utils/interfaceutil.h>
#include <memory>


namespace zeno
{
    ListObject::ListObject() : m_impl(std::make_unique<ListObject_impl>()) {
    }

    ListObject::ListObject(const ListObject& rhs) {
        m_impl = std::make_unique<ListObject_impl>(*rhs.m_impl);
        update_key(rhs.key());
    }

    zany ListObject::clone() const {
        return std::make_unique<ListObject>(*this);
    }

    void ListObject::Delete() {
    }

    void ListObject::clear() {
        m_impl->clear();
    }

    bool ListObject::has_change_info() const {
        return !m_impl->m_modify.empty() || !m_impl->m_new_added.empty() || !m_impl->m_new_removed.empty();
    }

    void ListObject::update_key(const String& key) {
        if (key.empty()) return;

        base::update_key(key);
        for (int i = 0; i < m_impl->m_objects.size(); i++) {
            auto& obj = m_impl->m_objects[i];
            if (obj->key().empty()) {
                std::string newkey = zsString2Std(m_key) + "/" + std::to_string(i);
                obj->update_key(stdString2zs(newkey));
            }
        }
    }

    ListObject::~ListObject() {
        m_impl->remove_children();
    }

    size_t ListObject::size() {
        return m_impl->size();
    }

    IObject* ListObject::get(int index) {
        return m_impl->get(index);
    }

    zeno::Vector<IObject*> ListObject::get() {
        std::vector<IObject*> v = m_impl->get();
        zeno::Vector<IObject*> vec(v.size());
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = v[i];
        }
        return vec;
    }

    void ListObject::push_back(zany&& obj) {
        m_impl->push_back(std::move(obj));
    }

    //void ListObject::set(const zeno::Vector<zany>& arr) {
    //    m_impl->set(zeVec2stdVec(arr));
    //}

    void ListObject::set(size_t index, zany&& obj) {
        m_impl->set(index, std::move(obj));
    }



    ListObject_impl::ListObject_impl(const ListObject_impl& listobj) {
        dirtyIndice = listobj.dirtyIndice;

        m_objects.resize(listobj.size());
        for (int i = 0; i < listobj.size(); ++i) {
            m_objects[i] = listobj.get(i)->clone();
            m_objects[i]->update_key(listobj.get(i)->key());
        }

        m_modify = listobj.m_modify;
        m_new_added = listobj.m_new_added;
        m_new_removed = listobj.m_new_removed;
    }

    ListObject_impl::ListObject_impl(const std::vector<zany>& arrin) {
        set(arrin);
    }

    void ListObject_impl::set(const std::vector<zany>& arr) {
        for (auto& obj : arr) {
            m_objects.push_back(obj->clone());
        }
    }

    void ListObject_impl::remove_children() {
        m_objects.clear();
        //for (auto pObject : m_objects) {
        //    pObject->Delete();
        //}
    }

    std::vector<std::string> ListObject_impl::paths() const {
        std::vector<std::string> _paths;
        for (const auto& obj : m_objects) {
            std::vector<std::string> subpaths = get_obj_paths(obj.get());
            for (const std::string& subpath : subpaths) {
                _paths.push_back(subpath);
            }
        }
        return _paths;
    }

    void ListObject_impl::resize(const size_t sz) {
        m_objects.resize(sz);
    }

    void ListObject_impl::append(zany&& spObj) {
        m_objects.push_back(std::move(spObj));
    }

    IObject* ListObject_impl::get(int index) const {
        if (0 > index || index >= m_objects.size())
            return nullptr;
        return m_objects[index].get();
    }

    void ListObject_impl::set(size_t index, zany&& obj) {
        if (0 > index || index >= m_objects.size())
            return;
        m_objects[index] = std::move(obj);
    }

    void ListObject_impl::mark_dirty(int index) {
        dirtyIndice.insert(index);
    }

    bool ListObject_impl::has_dirty(int index) const {
        return dirtyIndice.count(index);
    }

    size_t ListObject_impl::size() const {
        return m_objects.size();
    }

    size_t ListObject_impl::dirtyIndiceSize() {
        return dirtyIndice.size();
    }

    void ListObject_impl::emplace_back(zany&& obj) {
        append(std::move(obj));
    }

    void ListObject_impl::push_back(zany&& obj) {
        append(std::move(obj));
    }

    void ListObject_impl::clear() {
        m_new_added.clear();
        m_modify.clear();
        m_new_removed.clear();
        m_objects.clear();
    }

    std::unique_ptr<ListObject> create_ListObject() {
        return std::make_unique<ListObject>();
    }

    std::unique_ptr<ListObject> create_ListObject(std::vector<zany>&& arrin) {
        auto pList = std::make_unique<ListObject>();
        pList->m_impl = std::make_unique<ListObject_impl>(arrin);
        return pList;
    }
}