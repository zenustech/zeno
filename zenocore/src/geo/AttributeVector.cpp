#include <zeno/types/AttributeVector.h>

#define SHARED_COLUMN_DATA

namespace zeno {

    static bool sigval_init = false; //是否用单值优化：一列相同的数据用一个数值表达,TODO:copy没支持

    ZENO_API AttributeVector::AttributeVector(const AttributeVector& rhs)
#ifdef SHARED_COLUMN_DATA
        : x_comp(rhs.x_comp)
        , y_comp(rhs.y_comp)
        , z_comp(rhs.z_comp)
        , w_comp(rhs.w_comp)
        , m_size(rhs.m_size)
        , m_type(rhs.m_type)
        , self(rhs.self)
#endif
    {
#ifndef SHARED_COLUMN_DATA
        if (rhs.x_comp) x_comp = std::make_shared<AttrColumn>(*rhs.x_comp);
        if (rhs.y_comp) y_comp = std::make_shared<AttrColumn>(*rhs.y_comp);
        if (rhs.z_comp) z_comp = std::make_shared<AttrColumn>(*rhs.z_comp);
        if (rhs.w_comp) w_comp = std::make_shared<AttrColumn>(*rhs.w_comp);
        if (rhs.self) self = std::make_shared<AttrColumn>(*rhs.self);
        m_size = rhs.m_size;
        m_type = rhs.m_type;
#endif
    }

    ZENO_API AttributeVector::AttributeVector(const AttrVar& val_or_vec, size_t size, bool sigval_init)
        : m_size(size)
        , m_type(ATTR_TYPE_UNKNOWN)
    {
        set(val_or_vec, sigval_init);
    }

    ZENO_API size_t AttributeVector::size() const {
        return m_size;
    }

    ZENO_API GeoAttrType AttributeVector::type() const {
        return m_type;
    }

    ZENO_API AttrValue AttributeVector::getelem(size_t idx) const {
        if (self)
            return self->get_elem(idx);
        switch (m_type)
        {
        case ATTR_INT:
        case ATTR_FLOAT:
        case ATTR_STRING:
            return self->get_elem(idx);
        case ATTR_VEC2:
        {
            vec2f vec(std::get<float>(x_comp->get_elem(idx)),
                std::get<float>(y_comp->get_elem(idx)));
            return vec;
        }
        case ATTR_VEC3:
        {
            vec3f vec(std::get<float>(x_comp->get_elem(idx)),
                std::get<float>(y_comp->get_elem(idx)),
                std::get<float>(z_comp->get_elem(idx)));
            return vec;
        }
        case ATTR_VEC4:
        {
            vec4f vec(std::get<float>(x_comp->get_elem(idx)),
                std::get<float>(y_comp->get_elem(idx)),
                std::get<float>(z_comp->get_elem(idx)),
                std::get<float>(w_comp->get_elem(idx)));
            return vec;
        }
        }
        assert(false);
        return AttrValue();
    }

    ZENO_API AttrValue AttributeVector::front() const {
        if (self)
            return self->front();
        switch (m_type)
        {
        case ATTR_INT:
        case ATTR_FLOAT:
        case ATTR_STRING:
            return self->front();
        case ATTR_VEC2:
        {
            vec2f vec(std::get<float>(x_comp->front()), 
                std::get<float>(y_comp->front()));
            return vec;
        }
        case ATTR_VEC3:
        {
            vec3f vec(std::get<float>(x_comp->front()),
                std::get<float>(y_comp->front()),
                std::get<float>(z_comp->front()));
            return vec;
        }
        case ATTR_VEC4:
        {
            vec4f vec(std::get<float>(x_comp->front()),
                std::get<float>(y_comp->front()),
                std::get<float>(z_comp->front()),
                std::get<float>(w_comp->front()));
            return vec;
        }
        }
        assert(false);
        return AttrValue();
    }

    ZENO_API void AttributeVector::copySlice(const AttributeVector& rhs, int fromIndex) {
        if (x_comp) {
            x_comp->copy(*rhs.x_comp, fromIndex);
        }
        if (y_comp) {
            y_comp->copy(*rhs.y_comp, fromIndex);
        }
        if (z_comp) {
            z_comp->copy(*rhs.z_comp, fromIndex);
        }
        if (w_comp) {
            w_comp->copy(*rhs.w_comp, fromIndex);
        }
        if (self) {
            self->copy(*rhs.self, fromIndex);
        }
    }

    ZENO_API void AttributeVector::set(const AttrVar& val_or_vec, bool) {
        std::visit([&](auto&& val) {
            using E = std::decay_t<decltype(val)>;

            if constexpr (std::is_same_v<E, int>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_INT && m_type != ATTR_FLOAT)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                self = std::make_shared<AttrColumn>(std::vector<E>(N_vec, val), m_size);
                m_type = ATTR_INT;
            }
            else if constexpr (std::is_same_v<E, float>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_INT && m_type != ATTR_FLOAT)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                self = std::make_shared<AttrColumn>(std::vector<E>(N_vec, val), m_size);
                m_type = ATTR_FLOAT;
            }
            else if constexpr (std::is_same_v<E, std::string>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_STRING)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                self = std::make_shared<AttrColumn>(std::vector<E>(N_vec, val), m_size);
                m_type = ATTR_STRING;
            }
            else if constexpr (std::is_same_v<E, vec2i> || std::is_same_v<E, vec2f> || std::is_same_v<E, glm::vec2>) {
                //只能以float储存
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC2)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                x_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[0]), m_size);
                y_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[0]), m_size);
                m_type = ATTR_VEC2;
            }
            else if constexpr (std::is_same_v<E, vec3i> || std::is_same_v<E, vec3f> || std::is_same_v<E, glm::vec3>) {
                //只能以float储存
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC3)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                x_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[0]), m_size);
                y_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[1]), m_size);
                z_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[2]), m_size);
                m_type = ATTR_VEC3;
            }
            else if constexpr (std::is_same_v<E, vec4i> || std::is_same_v<E, vec4f> || std::is_same_v<E, glm::vec4>) {
                //只能以float储存
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC4)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                const int N_vec = sigval_init ? 1 : m_size;
                x_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[0]), m_size);
                y_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[1]), m_size);
                z_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[2]), m_size);
                w_comp = std::make_shared<AttrColumn>(std::vector<float>(N_vec, val[3]), m_size);
                m_type = ATTR_VEC4;
            }
            else if constexpr (std::is_same_v<E, std::vector<int>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_INT && m_type != ATTR_FLOAT)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                self = std::make_shared<AttrColumn>(val, m_size);
                m_type = ATTR_INT;
            }
            else if constexpr (std::is_same_v<E, std::vector<float>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_INT && m_type != ATTR_FLOAT)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                self = std::make_shared<AttrColumn>(val, m_size);
                m_type = ATTR_FLOAT;
            }
            else if constexpr (std::is_same_v<E, std::vector<std::string>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_STRING)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                self = std::make_shared<AttrColumn>(val, m_size);
                m_type = ATTR_STRING;
            }
            else if constexpr (std::is_same_v<E, std::vector<vec2i>> || std::is_same_v<E, std::vector<vec2f>> || std::is_same_v<E, std::vector<glm::vec2>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC2)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                std::vector<float> xvec(m_size);
                std::vector<float> yvec(m_size);
                for (size_t i = 0; i < m_size; i++) {
                    xvec[i] = val[i][0];
                    yvec[i] = val[i][1];
                }
                x_comp = std::make_shared<AttrColumn>(xvec, m_size);
                y_comp = std::make_shared<AttrColumn>(yvec, m_size);
                m_type = ATTR_VEC2;
            }
            else if constexpr (std::is_same_v<E, std::vector<vec3i>> || std::is_same_v<E, std::vector<vec3f>> || std::is_same_v<E, std::vector<glm::vec3>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC3)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                std::vector<float> xvec(m_size);
                std::vector<float> yvec(m_size);
                std::vector<float> zvec(m_size);
                for (size_t i = 0; i < m_size; i++) {
                    xvec[i] = val[i][0];
                    yvec[i] = val[i][1];
                    zvec[i] = val[i][2];
                }
                x_comp = std::make_shared<AttrColumn>(xvec, m_size);
                y_comp = std::make_shared<AttrColumn>(yvec, m_size);
                z_comp = std::make_shared<AttrColumn>(zvec, m_size);
                m_type = ATTR_VEC3;
            }
            else if constexpr (std::is_same_v<E, std::vector<vec4i>> || std::is_same_v<E, std::vector<vec4f>> || std::is_same_v<E, std::vector<glm::vec4>>) {
                if (m_type != ATTR_TYPE_UNKNOWN && m_type != ATTR_VEC4)
                    throw makeError<UnimplError>("type dismatch, cannot change type of attrvector");
                std::vector<float> xvec(m_size);
                std::vector<float> yvec(m_size);
                std::vector<float> zvec(m_size);
                std::vector<float> wvec(m_size);
                for (size_t i = 0; i < m_size; i++) {
                    xvec[i] = val[i][0];
                    yvec[i] = val[i][1];
                    zvec[i] = val[i][2];
                    wvec[i] = val[i][3];
                }
                x_comp = std::make_shared<AttrColumn>(xvec, m_size);
                y_comp = std::make_shared<AttrColumn>(yvec, m_size);
                z_comp = std::make_shared<AttrColumn>(zvec, m_size);
                w_comp = std::make_shared<AttrColumn>(wvec, m_size);
                m_type = ATTR_VEC4;
            }
        }, val_or_vec);
    }

#ifdef TRACE_GEOM_ATTR_DATA
    void AttributeVector::set_xcomp_id(const std::string& id) {
        if (x_comp)
            x_comp->_id = id;
    }

    void AttributeVector::set_ycomp_id(const std::string& id) {
        if (y_comp)
            y_comp->_id = id;
    }

    void AttributeVector::set_zcomp_id(const std::string& id) {
        if (z_comp)
            z_comp->_id = id;
    }

    void AttributeVector::set_self_id(const std::string& id) {
        if (self)
            self->_id = id;
    }

    std::string AttributeVector::xcomp_id() {
        return x_comp ? x_comp->_id : "";
    }

    std::string AttributeVector::ycomp_id() {
        return y_comp ? y_comp->_id : "";
    }

    std::string AttributeVector::zcomp_id() {
        return z_comp ? z_comp->_id : "";
    }

    std::string AttributeVector::wcomp_id() {
        return w_comp ? w_comp->_id : "";
    }

    std::string AttributeVector::self_id() {
        return self ? self->_id : "";
    }
#endif
}