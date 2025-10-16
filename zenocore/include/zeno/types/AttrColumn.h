#pragma once

#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/core/data.h>
#include <zeno/utils/Error.h>
#include <mutex>


namespace zeno {

    class AttributeImpl {
    public:
        void set(AttrVarOrVec elemVal) {
            m_data = elemVal;
        }

        template<typename T>
        void set(size_t index, T elemVal) {
            if (index < 0 || index >= m_size) {
                throw makeError<UnimplError>("index exceed the range of array");
            }

            std::visit([&](auto&& vec) {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (!std::is_same_v<E, std::vector<T>>) {
                    //TODO: 类型冲突时的处理
                    throw;
                }
                else if constexpr (std::is_same_v<E, std::vector<int>> ||
                        std::is_same_v<E, std::vector<float>> ||
                        std::is_same_v<E, std::vector<std::string>>) {
                    if (vec.size() == 1 && m_size > 1) {
                        vec.resize(m_size, vec[0]);
                    }
                    vec[index] = elemVal;
                }
            }, m_data);
        }

        template<typename T>
        T get(size_t index) const {
            return std::visit([&](auto&& vec)->T {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (!std::is_same_v<E, std::vector<T>>) {
                    throw makeError<UnimplError>("the type T you want to get doesn't match with the type of variant");
                }
                else if constexpr (std::is_same_v<E, std::vector<int>>) {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    return vec[index];
                }
                else if constexpr (std::is_same_v<E, std::vector<float>>) {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    return vec[index];
                }
                else if constexpr (std::is_same_v<E, std::vector<std::string>>) {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    return vec[index];
                }
                else {
                    throw;
                }
            }, m_data);
        }

        AttrVarOrVec get() const {
            return m_data;
        }

        AttrValue front() const {
            return std::visit([&](auto&& vec)->AttrValue {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (
                    std::is_same_v<E, std::vector<int>> ||
                    std::is_same_v < E, std::vector<float>> ||
                    std::is_same_v < E, std::vector<std::string>> ||
                    std::is_same_v<E, std::vector<vec2f>> ||
                    std::is_same_v<E, std::vector<vec3f>> ||
                    std::is_same_v<E, std::vector<vec4f>>) {
                    return vec.front();
                }
            }, m_data);
        }

        AttrValue get_elem(size_t idx) const {
            return std::visit([&](auto&& vec)->AttrValue {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (
                    std::is_same_v<E, std::vector<int>> ||
                    std::is_same_v < E, std::vector<float>> ||
                    std::is_same_v < E, std::vector<std::string>> ||
                    std::is_same_v<E, std::vector<vec2f>> ||
                    std::is_same_v<E, std::vector<vec3f>> ||
                    std::is_same_v<E, std::vector<vec4f>>) {
                    if (vec.size() == 1) {
                        return vec[0];
                    }
                    else {
                        return vec[idx];
                    }
                }
            }, m_data);
        }

        template<typename T>
        void insert(size_t index, T elemVal) {
            if (index < 0 || index >= m_size) {
                throw makeError<UnimplError>("index exceed the range of array");
            }

            std::visit([&](auto&& vec) {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (!std::is_same_v<E, std::vector<T>>) {
                    throw;
                }
                else if constexpr (std::is_same_v<E, std::vector<int>> ||
                    std::is_same_v < E, std::vector<float>> ||
                    std::is_same_v < E, std::vector<std::string>>) {
                    if (vec.size() == 1 && m_size > 1) {
                        vec.resize(m_size, vec[0]);
                    }
                    vec.insert(vec.begin() + index, elemVal);
                }
            }, m_data);
            m_size++;
        }

        template<typename T>
        void append(T elementVal) {
            std::visit([&](auto&& vec) {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (!std::is_same_v<E, std::vector<T>>) {
                    throw;
                } else if constexpr (std::is_same_v<E, std::vector<int>> ||
                    std::is_same_v<E, std::vector<float>> ||
                    std::is_same_v<E, std::vector<std::string>>) {
                    if (vec.size() == 1 && m_size > 1) {
                        vec.resize(m_size, vec[0]);
                    }
                    vec.push_back(elementVal);
                }
            }, m_data);
            m_size++;
        }

        void remove(size_t index) {
            if (index < 0 || index >= m_size) {
                throw makeError<UnimplError>("index exceed the range of attribute data");
            }
            std::visit([&](auto&& vec) {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (std::is_same_v<E, std::vector<int>> ||
                    std::is_same_v < E, std::vector<float>> ||
                    std::is_same_v < E, std::vector<std::string>>) {
                    if (vec.size() == 1) {
                        return;
                    }
                    vec.erase(vec.begin() + index);
                }
            }, m_data);
            m_size--;
        }

        void copy(const AttributeImpl& rattr, int fromIndex) {
            //将rattr整个vector的值，拷贝到m_data[fromIndex:]，特供给merge拷贝属性的时候用
            if (rattr.m_data.index() != m_data.index()) {
                throw makeError<UnimplError>("type dismatch when copy");
            }
            std::visit([&](auto&& vec) {
                using E = std::decay_t<decltype(vec)>;
                if constexpr (std::is_same_v<E, std::vector<int>>) {
                    auto& rvec = std::get<std::vector<int>>(rattr.m_data);
                    if (vec.size() < rvec.size() + fromIndex) {
                        throw makeError<UnimplError>("data range exceeds when copy");
                    }
                    std::copy(rvec.begin(), rvec.end(), vec.begin() + fromIndex);
                }
                else if constexpr (std::is_same_v<E, std::vector<float>>) {
                    auto& rvec = std::get<std::vector<float>>(rattr.m_data);
                    if (vec.size() < rvec.size() + fromIndex) {
                        throw makeError<UnimplError>("data range exceeds when copy");
                    }
                    std::copy(rvec.begin(), rvec.end(), vec.begin() + fromIndex);
                }
                else if constexpr (std::is_same_v<E, std::vector<std::string>>) {
                    auto& rvec = std::get<std::vector<std::string>>(rattr.m_data);
                    if (vec.size() < rvec.size() + fromIndex) {
                        throw makeError<UnimplError>("data range exceeds when copy");
                    }
                    std::copy(rvec.begin(), rvec.end(), vec.begin() + fromIndex);
                }
                //目前向量是独立通道储存的，所以先不写vec类型
                else {
                    throw makeError<UnimplError>("unknown type in attrcolumn");
                }
            }, m_data);
        }

        size_t m_size;      //代表向量的大小，并不是向量实际的大小（可能是单值）
        //暂时不储存核心类型，不过有可能出现类型错误赋值的情况
        AttrVarOrVec m_data;
    };

    class AttrColumn {
    public:
        ZENO_API AttrColumn() = delete;
        ZENO_API AttrColumn(const AttrColumn& rhs);
        ZENO_API AttrColumn(AttrVarOrVec value, size_t size);
        ZENO_API ~AttrColumn();
        ZENO_API AttrVarOrVec& value() const;

        static std::shared_ptr<AttrColumn> copy_on_write(
            const std::shared_ptr<AttrColumn>& pColumn)
        {
            std::lock_guard lck(pColumn->m_mutex);
            if (pColumn.use_count() > 1) {
                return std::make_shared<AttrColumn>(*pColumn);
            }
            return pColumn;
        }

        template<typename T>
        T get(size_t index) const {
            return m_pImpl->get<T>(index);
        }

        AttrValue get_elem(size_t idx) const;
        AttrValue front() const;

        template<typename T>
        void set(size_t index, T val) {
            m_pImpl->set(index, val);
        }

        void set(const AttrVarOrVec& val);

        template<typename T>
        void insert(size_t index, T val) {
            m_pImpl->insert(index, val);
        }

        void copy(const AttrColumn& rCol, int fromIndex) {
            m_pImpl->copy(*rCol.m_pImpl, fromIndex);
        }

        template<typename T>
        void append(T val) {
            m_pImpl->append(val);
        }

        void remove(size_t index);
        size_t size() const;
#ifdef TRACE_GEOM_ATTR_DATA
        std::string _id;
#endif

    private:
         std::unique_ptr<AttributeImpl> m_pImpl;
         //目前的设定里，竞态条件只会在copy_on_write时发生，而不会发生，某线程在读
         //另一个线程在写的情况，对于后者，必须要copy_on_write出一份新的实例，此时读写已经
         //不在同一个对象了。
         //问题是：读线程和copy线程之间是否会发生数据竞争？
         //copy只是把底层的variant数据拷一下，不会修改数据本身，而读线程也只是取数据，看起来不会有
         //数据竞争
         std::mutex m_mutex;
    };

}