#include "reflect/utils/assert"
#include "reflect/container/string"
#include "reflect/container/arraylist"
#include "metadata.hpp"

#include <string_view>
#include <map>
#include <string>

using namespace zeno::reflect;

class StringMetadataValue : public IMetadataValue {
public:
    StringMetadataValue(StringView value) : m_value(zeno::reflect::move(value)) {}

protected:
    StringView m_value;

    virtual bool is_string() const override {
        return true;
    }

    virtual const char* as_string() const override {
        return m_value.data();
    }
};

class HeterogeneousListMetadataValue : public IMetadataValue {
public:
    template <typename... Args>
    HeterogeneousListMetadataValue(Args&&... args) : m_value_list({ args... }) {
    }

protected:
    ArrayList<UniquePtr<IMetadataValue>> m_value_list;

    virtual bool is_list() const override {
        return true;
    }

    virtual size_t list_length() const override {
        return m_value_list.size();
    }

    virtual const IMetadataValue* list_get_item(size_t index) const override {
        ZENO_CHECK_MSG(m_value_list.is_valid_index(index), "Invalid metadata value index");
        return m_value_list[index].get();
    }

    virtual void list_add_item(UniquePtr<IMetadataValue>&& value) override {
        m_value_list.add_item(zeno::reflect::forward<UniquePtr<IMetadataValue>>(value));
    }

    friend class RawMetadata;
};

class RawMetadata : public IRawMetadata {
public:
    RawMetadata() {}

    template <typename T>
    RawMetadata(std::initializer_list<T> init) : m_root_value(zeno::reflect::move(init)) {}

private:
    std::map<std::string, UniquePtr<zeno::reflect::IMetadataValue>> m_root_value;

protected:
    virtual const IMetadataValue* get_value(const char* key) const override {
        if (auto it = m_root_value.find(key); it !=  m_root_value.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    virtual void set_value(const char* key, zeno::reflect::UniquePtr<zeno::reflect::IMetadataValue>&& value) override {
        m_root_value.insert_or_assign(key, zeno::reflect::forward<zeno::reflect::UniquePtr<zeno::reflect::IMetadataValue>>(value));
    }
};


UniquePtr<IRawMetadata> zeno::reflect::IRawMetadata::create()
{
    return make_unique<RawMetadata>();
}

zeno::reflect::IRawMetadata::~IRawMetadata()
{
}

UniquePtr<IMetadataValue> zeno::reflect::IMetadataValue::create_string(const char *str)
{
    return make_unique<StringMetadataValue>(str);
}

UniquePtr<IMetadataValue> zeno::reflect::IMetadataValue::create_list()
{
    return make_unique<HeterogeneousListMetadataValue>();
}

zeno::reflect::IMetadataValue::~IMetadataValue()
{
}

bool zeno::reflect::IMetadataValue::is_string() const
{
    return false;
}

const char* zeno::reflect::IMetadataValue::as_string() const
{
    ZENO_CHECK_MSG(false, "This value isn't a string");
    return nullptr;
}

bool zeno::reflect::IMetadataValue::is_list() const
{
    return false;
}

size_t zeno::reflect::IMetadataValue::list_length() const
{
    ZENO_CHECK_MSG(false, "This value isn't a list");
    return 0;
}

const IMetadataValue *zeno::reflect::IMetadataValue::list_get_item(size_t index) const
{
    ZENO_CHECK_MSG(false, "This value isn't a list");
    return nullptr;
}

void zeno::reflect::IMetadataValue::list_add_item(UniquePtr<IMetadataValue> &&value)
{
}
