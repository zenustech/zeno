#pragma once

#include <zeno/utils/api.h>
#include <zeno/core/IObject.h>

namespace zeno {

struct ListObject_impl;

struct ZENO_API ListObject : IObjectClone<ListObject> {
    typedef IObjectClone<ListObject> base;

    ListObject();
    ListObject(const ListObject& rhs);
    ~ListObject();

    std::unique_ptr<IObject> clone() const override;
    void Delete() override;
    size_t size();
    void clear();
    IObject* get(int index);
    zeno::Vector<IObject*> get();
    void push_back(zany&& obj);
    //void set(const zeno::Vector<zany>& arr);
    void set(size_t index, zany&& obj);
    void update_key(const String& key) override;
    bool has_change_info() const;
    std::unique_ptr<ListObject_impl> m_impl;
};

ZENO_API std::unique_ptr<ListObject> create_ListObject();
ZENO_API std::unique_ptr<ListObject> create_ListObject(std::vector<zany>&& arrin);

}
