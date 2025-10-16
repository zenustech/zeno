#pragma once

#include <zeno/core/common.h>
#include <zeno/core/IObject.h>
#include <zeno/core/data.h>
#include <zeno/types/AttrColumn.h>
#include <zeno/types/PrimitiveObject.h>


namespace zeno {

    using namespace zeno::reflect;

    class AttributeVector
    {
    public:
        AttributeVector() = delete;
        ZENO_API AttributeVector(const AttrVar& val_or_vec, size_t size, bool sigval_init = true);
        ZENO_API AttributeVector(const AttributeVector& rhs);
        ZENO_API size_t size() const;
        ZENO_API void set(const AttrVar& val_or_vec, bool sigval_init = true);
        ZENO_API GeoAttrType type() const;
        ZENO_API AttrValue front() const;
        ZENO_API AttrValue getelem(size_t idx) const;
        ZENO_API void copySlice(const AttributeVector& rhs, int fromIndex);

#ifdef TRACE_GEOM_ATTR_DATA
        void set_xcomp_id(const std::string& id);
        void set_ycomp_id(const std::string& id);
        void set_zcomp_id(const std::string& id);
        void set_self_id(const std::string& id);
        std::string xcomp_id();
        std::string ycomp_id();
        std::string zcomp_id();
        std::string wcomp_id();
        std::string self_id();
#endif

        void to_prim_attr(PrimitiveObject* spPrim, bool is_point_attr, bool is_triangle, std::string const& attr_name) {
            if (self) {
                auto& valvar = self->value();

                std::visit([&](auto&& vec) {
                    using E = std::decay_t<decltype(vec)>;
                    if constexpr (std::is_same_v<E, std::vector<int>>) {
                        if (is_point_attr) {
                            auto& vec_attr = spPrim->verts.add_attr<int>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<int>(m_size, vec[0]) : vec;
                        }
                        else if (is_triangle){
                            auto& vec_attr = spPrim->tris.add_attr<int>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<int>(m_size, vec[0]) : vec;
                        }
                        else {
                            auto& vec_attr = spPrim->polys.add_attr<int>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<int>(m_size, vec[0]) : vec;
                        }
                    }
                    else if constexpr (std::is_same_v<E, std::vector<float>>) {
                        if (is_point_attr) {
                            auto& vec_attr = spPrim->verts.add_attr<float>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<float>(m_size, vec[0]) : vec;
                        }
                        else if (is_triangle) {
                            auto& vec_attr = spPrim->tris.add_attr<float>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<float>(m_size, vec[0]) : vec;
                        }
                        else {
                            auto& vec_attr = spPrim->polys.add_attr<float>(attr_name);
                            vec_attr = (vec.size() == 1) ? std::vector<float>(m_size, vec[0]) : vec;
                        }
                    }
                    else if constexpr (std::is_same_v < E, std::vector<std::string>>) {
                        // do not support string type on primitive object.
                    }
                    else {
                        throw;
                    }
                    }, valvar);
            }
            else if (x_comp && y_comp) {
                auto& xvar = x_comp->value();
                auto& yvar = y_comp->value();
                assert(xvar.index() == yvar.index());
                if (z_comp) {
                    auto& zvar = z_comp->value();
                    assert(zvar.index() == xvar.index());
                    if (w_comp) {
                        auto& wvar = w_comp->value();
                        assert(wvar.index() == xvar.index());

                        if (is_point_attr) {
                            auto& vec_attr = spPrim->verts.add_attr<zeno::vec4f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec4f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec4f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i), w_comp->get<float>(i));
                            }
                        }
                        else if (is_triangle) {
                            auto& vec_attr = spPrim->tris.add_attr<zeno::vec4f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec4f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec4f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i), w_comp->get<float>(i));
                            }
                        }
                        else {
                            auto& vec_attr = spPrim->polys.add_attr<zeno::vec4f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec4f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec4f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i), w_comp->get<float>(i));
                            }
                        }
                    }
                    else {
                        if (is_point_attr) {
                            auto& vec_attr = spPrim->verts.add_attr<zeno::vec3f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec3f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec3f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i));
                            }
                        }
                        else if (is_triangle) {
                            auto& vec_attr = spPrim->tris.add_attr<zeno::vec3f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec3f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec3f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i));
                            }
                        }
                        else {
                            auto& vec_attr = spPrim->polys.add_attr<zeno::vec3f>(attr_name);
                            vec_attr.resize(m_size, zeno::vec3f());
                            for (int i = 0; i < m_size; i++) {
                                vec_attr[i] = zeno::vec3f(x_comp->get<float>(i), y_comp->get<float>(i), z_comp->get<float>(i));
                            }
                        }
                    }
                }
                else {
                    if (is_point_attr) {
                        auto& vec_attr = spPrim->verts.add_attr<zeno::vec2f>(attr_name);
                        vec_attr.resize(m_size, zeno::vec2f());
                        for (int i = 0; i < m_size; i++) {
                            vec_attr[i] = zeno::vec2f(x_comp->get<float>(i), y_comp->get<float>(i));
                        }
                    }
                    else if (is_triangle) {
                        auto& vec_attr = spPrim->tris.add_attr<zeno::vec2f>(attr_name);
                        vec_attr.resize(m_size, zeno::vec2f());
                        for (int i = 0; i < m_size; i++) {
                            vec_attr[i] = zeno::vec2f(x_comp->get<float>(i), y_comp->get<float>(i));
                        }
                    }
                    else {
                        auto& vec_attr = spPrim->polys.add_attr<zeno::vec2f>(attr_name);
                        vec_attr.resize(m_size, zeno::vec2f());
                        for (int i = 0; i < m_size; i++) {
                            vec_attr[i] = zeno::vec2f(x_comp->get<float>(i), y_comp->get<float>(i));
                        }
                    }
                }
            }
            else {
                throw makeError<UnimplError>("internal data error on attribute vector");
            }
        }

        template<class T, char CHANNEL = 0>
        std::vector<T> get_attrs() const {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2>) {
                std::vector<T> vec(m_size);
                for (int i = 0; i < m_size; i++) {
                    float xval = x_comp->get<float>(i);
                    float yval = y_comp->get<float>(i);
                    vec[i] = T(xval, yval);
                }
                return vec;
            }
            else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                std::vector<T> vec(m_size);
                for (int i = 0; i < m_size; i++) {
                    float xval = x_comp->get<float>(i);
                    float yval = y_comp->get<float>(i);
                    float zval = z_comp->get<float>(i);
                    vec[i] = T(xval, yval, zval);
                }
                return vec;
            }
            else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                std::vector<T> vec(m_size);
                for (int i = 0; i < m_size; i++) {
                    float xval = x_comp->get<float>(i);
                    float yval = y_comp->get<float>(i);
                    float zval = z_comp->get<float>(i);
                    float wval = w_comp->get<float>(i);
                    vec[i] = T(xval, yval, zval, wval);
                }
                return vec;
            }
            else {
                if constexpr (CHANNEL > 0) {
                    std::shared_ptr<AttrColumn> spComp;
                    if constexpr (CHANNEL == 'x') { spComp = x_comp; }
                    else if constexpr (CHANNEL == 'y') { spComp = y_comp; }
                    else if constexpr (CHANNEL == 'z') { spComp = z_comp; }
                    else if constexpr (CHANNEL == 'w') { spComp = w_comp; }
                    if (!spComp)
                        throw makeError<UnimplError>("the variant doesn't contain component vector");

                    return std::visit([&](auto&& vec)->std::vector<T> {
                        using E = std::decay_t<decltype(vec)>;
                        if constexpr (std::is_same_v<E, std::vector<T>>) {
                            if (vec.size() == 1) {
                                return std::vector<T>(m_size, vec[0]);
                            }
                            else {
                                return vec;
                            }
                        }
                        else {
                            throw makeError<UnimplError>("type dismatch");
                        }
                    }, spComp->value());
                }
                else {
                    //现在对于向量采用分列储存，因此实质上内部只有int float string三种类型
                    auto& varvec = self->value();
                    return std::visit([&](auto&& vec)->std::vector<T> {
                        using E = std::decay_t<decltype(vec)>;
                        if constexpr (
                            (std::is_same_v<E, std::vector<int>> && std::is_same_v<T, int>) ||
                            (std::is_same_v<E, std::vector<float>> && std::is_same_v<T, float>) ||
                            (std::is_same_v<E, std::vector<std::string>> && std::is_same_v<T, std::string>)) {
                            if (vec.size() == 1) {
                                return std::vector<T>(m_size, vec[0]);
                            }
                            return vec;
                        }
                        else {
                            throw makeError<UnimplError>("internal type error of primitive type of attribute data");
                        }
                    }, varvec);
                }
            }
        }

        template<typename T>
        T get_elem(char channel, size_t idx) const {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2>) {
                return T(x_comp->get<float>(idx), y_comp->get<float>(idx));
            }
            else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                return T(x_comp->get<float>(idx), y_comp->get<float>(idx), z_comp->get<float>(idx));
            }
            else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                return T(x_comp->get<float>(idx), y_comp->get<float>(idx), z_comp->get<float>(idx), w_comp->get<float>(idx));
            }
            else {
                if (channel > 0) {
                    std::shared_ptr<AttrColumn> spComp;
                    if (channel == 'x') { spComp = x_comp; }
                    else if (channel == 'y') { spComp = y_comp; }
                    else if (channel == 'z') { spComp = z_comp; }
                    else if (channel == 'w') { spComp = w_comp; }
                    if (!spComp)
                        throw makeError<UnimplError>("the variant doesn't contain component vector");
                    return std::visit([&](auto&& vec)->T {
                        using E = std::decay_t<decltype(vec)>;
                        if constexpr (std::is_same_v<E, std::vector<T>>) {
                            if (vec.size() == 1) {
                                return vec[0];
                            }
                            else {
                                return vec[idx];
                            }
                        }
                        else {
                            throw;
                        }
                    }, spComp->value());
                }
                else {
                    AttrVarOrVec& attrval = self->value();
                    return self->get<T>(idx);
                }
            }
        }

        template<class T, char CHANNEL = 0>
        void set_elem(size_t idx, T val) {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                x_comp->set(idx, val[0]);
                y_comp->set(idx, val[1]);
            }
            else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                x_comp->set(idx, val[0]);
                y_comp->set(idx, val[1]);
                z_comp->set(idx, val[2]);
            }
            else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                w_comp = AttrColumn::copy_on_write(w_comp);
                x_comp->set(idx, val[0]);
                y_comp->set(idx, val[1]);
                z_comp->set(idx, val[2]);
                w_comp->set(idx, val[3]);
            }
            else {
                if constexpr (CHANNEL > 0) {
                    if constexpr (CHANNEL == 'x') {
                        x_comp = AttrColumn::copy_on_write(x_comp);
                        x_comp->set(idx, val);
                    }
                    else if constexpr (CHANNEL == 'y') {
                        y_comp = AttrColumn::copy_on_write(y_comp);
                        y_comp->set(idx, val);
                    }
                    else if constexpr (CHANNEL == 'z') {
                        z_comp = AttrColumn::copy_on_write(z_comp);
                        z_comp->set(idx, val);
                    }
                    else if constexpr (CHANNEL == 'w') {
                        w_comp = AttrColumn::copy_on_write(w_comp);
                        w_comp->set(idx, val);
                    }
                    else {
                        throw;
                    }
                }
                else {
                    self = AttrColumn::copy_on_write(self);
                    self->set(idx, val);
                }
            }
        }

        template<class T, char CHANNEL = 0>
        void insert_elem(size_t idx, T val) {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                x_comp->insert(idx, val[0]);
                y_comp->insert(idx, val[1]);
            } else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                x_comp->insert(idx, val[0]);
                y_comp->insert(idx, val[1]);
                z_comp->insert(idx, val[2]);
            } else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                w_comp = AttrColumn::copy_on_write(w_comp);
                x_comp->insert(idx, val[0]);
                y_comp->insert(idx, val[1]);
                z_comp->insert(idx, val[2]);
                w_comp->insert(idx, val[3]);
            } else {
                if constexpr (CHANNEL > 0) {
                    if constexpr (CHANNEL == 'x') {
                        x_comp = AttrColumn::copy_on_write(x_comp);
                        x_comp->insert(idx, val);
                    } else if constexpr (CHANNEL == 'y') {
                        y_comp = AttrColumn::copy_on_write(y_comp);
                        y_comp->insert(idx, val);
                    } else if constexpr (CHANNEL == 'z') {
                        z_comp = AttrColumn::copy_on_write(z_comp);
                        z_comp->insert(idx, val);
                    } else if constexpr (CHANNEL == 'w') {
                        w_comp = AttrColumn::copy_on_write(w_comp);
                        w_comp->insert(idx, val);
                    } else {
                        throw;
                    }
                } else {
                    self = AttrColumn::copy_on_write(self);
                    self->insert(idx, val);
                }
            }
            m_size++;
        };

        template<class T, char CHANNEL = 0>
        void append(T val) {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2> ) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                x_comp->append(val[0]);
                y_comp->append(val[1]);
            }
            else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                x_comp->append(val[0]);
                y_comp->append(val[1]);
                z_comp->append(val[2]);
            }
            else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                w_comp = AttrColumn::copy_on_write(w_comp);
                x_comp->append(val[0]);
                y_comp->append(val[1]);
                z_comp->append(val[2]);
                w_comp->append(val[3]);
            }
            else {
                if constexpr (CHANNEL > 0) {
                    if constexpr (CHANNEL == 'x') {
                        x_comp = AttrColumn::copy_on_write(x_comp);
                        x_comp->append(val);
                    } else if constexpr (CHANNEL == 'y') {
                        y_comp = AttrColumn::copy_on_write(y_comp);
                        y_comp->append(val);
                    } else if constexpr (CHANNEL == 'z') {
                        z_comp = AttrColumn::copy_on_write(z_comp);
                        z_comp->append(val);
                    } else if constexpr (CHANNEL == 'w') {
                        w_comp = AttrColumn::copy_on_write(w_comp);
                        w_comp->append(val);
                    } else {
                        throw;
                    }
                } else {
                    self = AttrColumn::copy_on_write(self);
                    self->append(val);
                }
            }
            m_size++;
        };

        template<class T, char CHANNEL = 0>
        void remove_elem(size_t idx) {
            if constexpr (std::is_same_v<T, vec2f> || std::is_same_v<T, glm::vec2>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                x_comp->remove(idx);
                y_comp->remove(idx);
            } else if constexpr (std::is_same_v<T, vec3f> || std::is_same_v<T, glm::vec3>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                x_comp->remove(idx);
                y_comp->remove(idx);
                z_comp->remove(idx);
            } else if constexpr (std::is_same_v<T, vec4f> || std::is_same_v<T, glm::vec4>) {
                x_comp = AttrColumn::copy_on_write(x_comp);
                y_comp = AttrColumn::copy_on_write(y_comp);
                z_comp = AttrColumn::copy_on_write(z_comp);
                w_comp = AttrColumn::copy_on_write(w_comp);
                x_comp->remove(idx);
                y_comp->remove(idx);
                z_comp->remove(idx);
                w_comp->remove(idx);
            } else {
                if constexpr (CHANNEL > 0) {
                    if constexpr (CHANNEL == 'x') {
                        x_comp = AttrColumn::copy_on_write(x_comp);
                        x_comp->remove(idx);
                    }
                    else if constexpr (CHANNEL == 'y') {
                        y_comp = AttrColumn::copy_on_write(y_comp);
                        y_comp->remove(idx);
                    }
                    else if constexpr (CHANNEL == 'z') {
                        z_comp = AttrColumn::copy_on_write(z_comp);
                        z_comp->remove(idx);
                    }
                    else if constexpr (CHANNEL == 'w') {
                        w_comp = AttrColumn::copy_on_write(w_comp);
                        w_comp->remove(idx);
                    }
                    else {
                        throw;
                    }
                } else {
                    self = AttrColumn::copy_on_write(self);
                    self->remove(idx);
                }
            }
            m_size--;
        };

        template<class T>
        void foreach_attr_update(char channel, std::function<T(int idx, T old_elem_value)>&& evalf) {
            if (channel > 0) {
                std::shared_ptr<AttrColumn> spComp;
                if (channel == 'x') {
                    x_comp = AttrColumn::copy_on_write(x_comp);
                    spComp = x_comp;
                }
                else if (channel == 'y') {
                    y_comp = AttrColumn::copy_on_write(y_comp);
                    spComp = y_comp;
                }
                else if (channel == 'z') {
                    z_comp = AttrColumn::copy_on_write(z_comp);
                    spComp = z_comp;
                }
                else if (channel == 'w') {
                    w_comp = AttrColumn::copy_on_write(w_comp);
                    spComp = w_comp;
                }
                else {
                    throw makeError<UnimplError>("Unknown channel");
                }

                AttrVarOrVec& var = spComp->value();
                std::visit([&](auto& vec) {
                    using E = std::decay_t<decltype(vec)>;
                    if constexpr (std::is_same_v<std::vector<T>, E>) {
                        int sz = vec.size();
                        if (sz != m_size) {
                            vec.resize(m_size);
                            sz = m_size;
                        }
                        for (int i = 0; i < m_size; i++) {
                            int idx = std::min(i, sz - 1);
                            T old_val(vec[idx]);
                            T new_val = evalf(i, old_val);
                            vec[idx] = new_val;
                        }
                    }
                    else {
                        throw makeError<UnimplError>("type dismatch when visit attribute value and modify them");
                    }
                }, var);
            }
            else {
                if constexpr (std::is_same_v<T, vec2f>) {
                    x_comp = AttrColumn::copy_on_write(x_comp);
                    y_comp = AttrColumn::copy_on_write(y_comp);
                    AttrVarOrVec& xvar = x_comp->value();
                    AttrVarOrVec& yvar = y_comp->value();
                    auto pXVec = std::get_if<std::vector<float>>(&xvar);
                    auto pYVec = std::get_if<std::vector<float>>(&yvar);
                    if (!pXVec || !pYVec)
                        throw makeError<UnimplError>("type dismatch");
                    int nx = pXVec->size(), ny = pYVec->size();
                    if (nx != m_size) {
                        nx = m_size;
                        pXVec->resize(m_size);
                    }
                    if (ny != m_size) {
                        ny = m_size;
                        pYVec->resize(m_size);
                    }
#pragma omp parallel for
                    for (int i = 0; i < m_size; i++) {
                        int ix = std::min(i, nx - 1), iy = std::min(i, ny - 1);
                        T old_val((*pXVec)[ix], (*pYVec)[iy]);
                        T new_val = evalf(i, old_val);
                        (*pXVec)[ix] = new_val[0];
                        (*pYVec)[iy] = new_val[1];
                    }
                }
                else if constexpr (std::is_same_v<T, vec3f>) {
                    x_comp = AttrColumn::copy_on_write(x_comp);
                    y_comp = AttrColumn::copy_on_write(y_comp);
                    z_comp = AttrColumn::copy_on_write(z_comp);

                    AttrVarOrVec& xvar = x_comp->value();
                    AttrVarOrVec& yvar = y_comp->value();
                    AttrVarOrVec& zvar = z_comp->value();
                    auto pXVec = std::get_if<std::vector<float>>(&xvar);
                    auto pYVec = std::get_if<std::vector<float>>(&yvar);
                    auto pZVec = std::get_if<std::vector<float>>(&zvar);
                    if (!pXVec || !pYVec || !pZVec)
                        throw makeError<UnimplError>("type dismatch");
                    int nx = pXVec->size(), ny = pYVec->size(), nz = pZVec->size();
                    if (nx != m_size) {
                        nx = m_size;
                        pXVec->resize(m_size);
                    }
                    if (ny != m_size) {
                        ny = m_size;
                        pYVec->resize(m_size);
                    }
                    if (nz != m_size) {
                        nz = m_size;
                        pZVec->resize(m_size);
                    }
#pragma omp parallel for
                    for (int i = 0; i < m_size; i++) {
                        int ix = std::min(i, nx - 1), iy = std::min(i, ny - 1), iz = std::min(i, nz - 1);
                        T old_val((*pXVec)[ix], (*pYVec)[iy], (*pZVec)[iz]);
                        T new_val = evalf(i, old_val);
                        (*pXVec)[ix] = new_val[0];
                        (*pYVec)[iy] = new_val[1];
                        (*pZVec)[iz] = new_val[2];
                    }
                }
                else if constexpr (std::is_same_v<T, vec4f>) {

                }
                else if constexpr (std::is_same_v<T, glm::vec2>) {
                    x_comp = AttrColumn::copy_on_write(x_comp);
                    y_comp = AttrColumn::copy_on_write(y_comp);
                    AttrVarOrVec& xvar = x_comp->value();
                    AttrVarOrVec& yvar = y_comp->value();
                    auto pXVec = std::get_if<std::vector<float>>(&xvar);
                    auto pYVec = std::get_if<std::vector<float>>(&yvar);
                    if (!pXVec || !pYVec)
                        throw makeError<UnimplError>("type dismatch");
                    int nx = pXVec->size(), ny = pYVec->size();
                    if (nx != m_size) {
                        nx = m_size;
                        pXVec->resize(m_size);
                    }
                    if (ny != m_size) {
                        ny = m_size;
                        pYVec->resize(m_size);
                    }
#pragma omp parallel for
                    for (int i = 0; i < m_size; i++) {
                        int ix = std::min(i, nx - 1), iy = std::min(i, ny - 1);
                        T old_val((*pXVec)[ix], (*pYVec)[iy]);
                        T new_val = evalf(i, old_val);
                        (*pXVec)[ix] = new_val[0];
                        (*pYVec)[iy] = new_val[1];
                    }
                }
                else if constexpr (std::is_same_v<T, glm::vec3>) {
                    x_comp = AttrColumn::copy_on_write(x_comp);
                    y_comp = AttrColumn::copy_on_write(y_comp);
                    z_comp = AttrColumn::copy_on_write(z_comp);

                    AttrVarOrVec& xvar = x_comp->value();
                    AttrVarOrVec& yvar = y_comp->value();
                    AttrVarOrVec& zvar = z_comp->value();
                    auto pXVec = std::get_if<std::vector<float>>(&xvar);
                    auto pYVec = std::get_if<std::vector<float>>(&yvar);
                    auto pZVec = std::get_if<std::vector<float>>(&zvar);
                    if (!pXVec || !pYVec || !pZVec)
                        throw makeError<UnimplError>("type dismatch");
                    int nx = pXVec->size(), ny = pYVec->size(), nz = pZVec->size();
                    if (nx != m_size) {
                        pXVec->resize(m_size);
                        nx = m_size;
                    }
                    if (ny != m_size) {
                        pYVec->resize(m_size);
                        ny = m_size;
                    }
                    if (nz != m_size) {
                        pZVec->resize(m_size);
                        nz = m_size;
                    }
#pragma omp parallel for
                    for (int i = 0; i < m_size; i++) {
                        int ix = std::min(i, nx - 1), iy = std::min(i, ny - 1), iz = std::min(i, nz - 1);
                        T old_val((*pXVec)[ix], (*pYVec)[iy], (*pZVec)[iz]);
                        T new_val = evalf(i, old_val);
                        (*pXVec)[ix] = new_val[0];
                        (*pYVec)[iy] = new_val[1];
                        (*pZVec)[iz] = new_val[2];
                    }
                }
                else if constexpr (std::is_same_v<T, glm::vec4>) {

                }
                else {
                    self = AttrColumn::copy_on_write(self);
                    AttrVarOrVec& selfvar = self->value();
                    std::visit([&](auto&& val) {
                        using E = std::decay_t<decltype(val)>;
                        int sz = val.size();
                        if (sz != m_size) {
                            val.resize(m_size);
                            sz = m_size;
                        }
                        if constexpr (std::is_same_v<E, std::vector<T>>) {
                            #pragma omp parallel for
                            for (int i = 0; i < m_size; i++) {
                                int ix = std::min(i, sz - 1);
                                T old_val(val[ix]);
                                T new_val = evalf(i, old_val);
                                val[ix] = new_val;
                            }
                        } else {
                            throw makeError<UnimplError>("type dismatch");
                        }
                    }, selfvar);
                }
            }
        }

    private:
        //这里把vector的分量拆出来，仅仅考虑了内存储存优化的情况，如果后续基于共享内存，写法会改变。
        std::shared_ptr<AttrColumn> x_comp, y_comp, z_comp, w_comp, self;
        size_t m_size;
        GeoAttrType m_type;
    };
}