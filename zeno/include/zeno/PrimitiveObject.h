#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <zeno/vec.h>
#include <zeno/zeno.h>
#include <zeno/memory.h>

namespace zeno {
struct PrimitiveAOS {
  //hey, interesting...
  //I found your implementation is still a SOA(structure of array)
  //I will do a real AOS here, since I think for wrangles, what we
  //need is AOS, for example m_arrs[i*start~i*start+start] gives all channels of particle i
    std::vector<float> m_arrs;
    
    std::map<std::string, int> channel_start;
    size_t m_size;
    int count = 0;

    // size_t nchannels() const {
    //     return m_chnames.size();
    // }

    // std::string channel_name(size_t i) const {
    //     return m_chnames[i];
    // }

    // void set_channel_name(size_t i, std::string const &name) {
    //     m_chnames[i] = name;
    // }
    void clear()
    {
      m_size = 0;
      count = 0;
      channel_start.clear();
      m_arrs.clear();
    }
    
    void setValue(int i, std::string channel,  zeno::vec3f a)
    {
      if(i>=0&&i<m_size)
      {
        m_arrs[i*count + channel_start[channel]]=a[0];
        m_arrs[i*count + channel_start[channel]+1]=a[1];
        m_arrs[i*count + channel_start[channel]+2]=a[2];
      }
      else if(i==-1)
      {
        channel_start[channel]=count;
        count+=3;
      }
    }
    void setValue(int i, std::string channel,  float a)
    {
      if(i>=0&&i<m_size)
      {
        m_arrs[i*count + channel_start[channel]]=a;
      }
      else if(i==-1)
      {
        channel_start[channel]=count;
        count+=1;
      }
    }
    template<class T>
    T getValue(int i, std::string channel)
    {
      if constexpr(std::is_same_v<T, float>)
      {
        return m_arrs[i*count + channel_start[channel]];
      }
      if constexpr(std::is_same_v<T, zeno::vec3f>)
      {
        return zeno::vec3f(m_arrs[i*count + channel_start[channel]],
        m_arrs[i*count + channel_start[channel]+1],
        m_arrs[i*count + channel_start[channel]+2]);
      }
    }
    
    template<class T>
    void countChannel(std::string channel, T value)
    {
      setValue(-1, channel, value);
    }
    float* getIndex(size_t i)
    {
      return &(m_arrs[i*count]);
    }
    int getChannelSize()
    {
      return count;
    }
    // size_t chid_of_name(std::string const &name) const {
    //     auto it = std::find(m_chnames.begin(), m_chnames.end(), name);
    //     return it - m_chnames.begin();
    // }

    // std::vector<float> &channel(size_t i) const {
    //     return *m_arrs[i].get();
    // }

    // void set_nchannels(size_t n) {
    //     size_t m = m_arrs.size();
    //     m_arrs.resize(n);
    //     m_chnames.resize(n);
    //     for (size_t i = m; i < n; i++) {
    //         m_arrs[i] = std::make_unique<std::vector<float>>(m_size);
    //     }
    //     for (size_t i = m; i < n; i++) {
    //         char buf[233];
    //         sprintf(buf, "ch%d", i);
    //         m_chnames[i] = buf;
    //     }
    // }

    size_t size() const {
        return m_size;
    }

    void resize(size_t num) {
        m_size = num;
        m_arrs.resize(num*count);
        
    }
};

using AttributeArray =
    std::variant<std::vector<zeno::vec3f>, std::vector<float>>;

struct PrimitiveObject : zeno::IObjectClone<PrimitiveObject> {

  std::map<std::string, AttributeArray> m_attrs;
  size_t m_size{0};

  std::vector<int> points;
  std::vector<zeno::vec2i> lines;
  std::vector<zeno::vec3i> tris;
  std::vector<zeno::vec4i> quads;

  PrimitiveAOS aosTwin;

  void toAOS()
  {
    aosTwin.clear();
    //count channels first
    for(auto &&a:m_attrs)
    {
      auto &&ch = a.first;
      if(a.second.index()==1)
        aosTwin.countChannel(ch, attr<float>(ch)[0]);
      if(a.second.index()==0)
        aosTwin.countChannel(ch, attr<zeno::vec3f>(ch)[0]);
    }
    aosTwin.resize(m_size);
    for(auto &&a:m_attrs)
    {
      if(a.second.index()==1){
        auto ch = a.first;
        #pragma omp parallel for
        for(size_t i=0;i<m_size;i++)
        {
          aosTwin.setValue(i, ch, std::get<std::vector<float>>(a.second)[i]);
        }
      }
      if(a.second.index()==0){
        auto ch = a.first;
        #pragma omp parallel for
        for(size_t i=0;i<m_size;i++)
        {
          aosTwin.setValue(i, ch, std::get<std::vector<zeno::vec3f>>(a.second)[i]);
        }
      }

    }

  }
  void toSOA()
  {
    if(aosTwin.size()>0)
    for(auto &&a:m_attrs)
    {
      if(a.second.index()==1){
        auto ch = a.first;
        #pragma omp parallel for
        for(size_t i=0;i<m_size;i++)
        {
          std::get<std::vector<float>>(a.second)[i] = aosTwin.getValue<float>(i, ch);
        }
      }
      if(a.second.index()==0){
        auto ch = a.first;
        #pragma omp parallel for
        for(size_t i=0;i<m_size;i++)
        {
          std::get<std::vector<zeno::vec3f>>(a.second)[i] = aosTwin.getValue<zeno::vec3f>(i, ch);
        }
      }

    }
  }

#ifndef ZEN_NOREFDLL
  ZENAPI virtual void dumpfile(std::string const &path) override;
#else
  virtual void dumpfile(std::string const &path) override {}
#endif

  template <class T> std::vector<T> &add_attr(std::string const &name) {
    if (!has_attr(name))
      m_attrs[name] = std::vector<T>(m_size);
    return attr<T>(name);
  }
  template <class T> std::vector<T> &add_attr(std::string const &name, T value) {
    if (!has_attr(name))
      m_attrs[name] = std::vector<T>(m_size, value);
    return attr<T>(name);
  }
  template <class T> std::vector<T> &attr(std::string const &name) {
    return std::get<std::vector<T>>(m_attrs.at(name));
  }

  AttributeArray &attr(std::string const &name) { return m_attrs.at(name); }

  template <class T> std::vector<T> const &attr(std::string const &name) const {
    return std::get<std::vector<T>>(m_attrs.at(name));
  }

  AttributeArray const &attr(std::string const &name) const {
    return m_attrs.at(name);
  }

  bool has_attr(std::string const &name) const {
    return m_attrs.find(name) != m_attrs.end();
  }

  template <class T> bool attr_is(std::string const &name) const {
    return std::holds_alternative<std::vector<T>>(m_attrs.at(name));
  }

  size_t size() const { return m_size; }

  void resize(size_t size) {
    m_size = size;
    for (auto &[key, val] : m_attrs) {
      std::visit([&](auto &val) { val.resize(m_size); }, val);
    }
  }
};


} // namespace zeno
