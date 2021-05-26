#pragma once

#include <zen/zen.h>
#include <zen/vec.h>
#include <variant>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace zenbase {

using AttributeArray = std::variant<
    std::vector<zen::vec3f>, std::vector<float>>;

inline bool is_vec3f_vector(const AttributeArray& v) {
    return v.index() == 0;
}

struct PrimitiveObject : zen::IObject {

    std::map<std::string, AttributeArray> m_attrs;
    size_t m_size{0};

    std::vector<int> points;
    std::vector<zen::vec2i> lines;
    std::vector<zen::vec3i> tris;
    std::vector<zen::vec4i> quads;

    template <class T>
    std::vector<T> &add_attr(std::string const &name) {
        if (!has_attr(name))
            m_attrs[name] = std::vector<T>(m_size);
        return attr<T>(name);
    }
    
    template <class T>
    T reduce(std::string channel, std::string type)
    {
        std::vector<T> temp = attr<T>(channel);
        
        if(type==std::string("avg")){
            T start=temp[0];
            auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                    start,
                    [&](tbb::blocked_range<int> r, T running_total)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total += temp[i];
                        }

                        return running_total;
                    }, [](auto a, auto b){return a+b; } );
            return total/(float)(temp.size());
        }
        if(type==std::string("max")){
            T start=temp[0];
            auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                    start,
                    [&](tbb::blocked_range<int> r, T running_total)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total = zen::max(running_total,temp[i]);
                        }

                        return running_total;
                    }, [](auto a, auto b) { return zen::max(a,b); } );
            return total;
        }
        if(type==std::string("min")){
            T start=temp[0];
            auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                    start,
                    [&](tbb::blocked_range<int> r, T running_total)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total = zen::min(running_total,temp[i]);
                        }

                        return running_total;
                    }, [](auto a, auto b) { return zen::min(a,b); } );
            return total;
        }
        if(type==std::string("absmax"))
        {
            T start=abs(temp[0]);
            auto total = tbb::parallel_reduce(tbb::blocked_range<int>(1,temp.size()),
                    start,
                    [&](tbb::blocked_range<int> r, T running_total)
                    {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total = zen::max(abs(running_total),abs(temp[i]));
                        }

                        return running_total;
                    }, [](auto a, auto b) { return zen::max(abs(a),abs(b)); } );
            return total;
        }
        
    }
    template <class T>
    std::vector<T> &attr(std::string const &name) {
        return std::get<std::vector<T>>(m_attrs.at(name));
    }

    AttributeArray &attr(std::string const &name) {
        return m_attrs.at(name);
    }

    template <class T>
    std::vector<T> const &attr(std::string const &name) const {
        return std::get<std::vector<T>>(m_attrs.at(name));
    }

    AttributeArray const &attr(std::string const &name) const {
        return m_attrs.at(name);
    }

    bool has_attr(std::string const &name) const {
        return m_attrs.find(name) != m_attrs.end();
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        return std::holds_alternative<std::vector<T>>(m_attrs.at(name));
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        m_size = size;
        for (auto &[key, val]: m_attrs) {
            std::visit([&](auto &val) {
                val.resize(m_size);
            }, val);
        }
        points.resize(m_size);
        #pragma omp parallel for
        for(int i=0;i<m_size;i++)
        {
            points[i] = i;
        }
    }
};

}
