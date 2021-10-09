#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/StringObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include <zeno/ZenoInc.h>

namespace zeno {

struct TraceOneStep : INode {
  virtual void apply() override {
    auto steps = get_input<NumericObject>("steps")->get<int>();
    auto prim = get_input<PrimitiveObject>("prim");
    auto vecField = get_input<VDBFloat3Grid>("vecField");
    auto size = get_input<NumericObject>("size")->get<int>();
    auto dt = get_input<NumericObject>("dt")->get<float>();
    auto maxlength = std::numeric_limits<float>::infinity();
    if(has_input("maxlength"))
    {
      maxlength = get_input<NumericObject>("maxlength")->get<float>();
    }
    
    for(auto s=0;s<steps;s++){
      prim->resize(prim->size()+size);
      auto &pos = prim->attr<vec3f>("pos");
      auto &lengtharr = prim->attr<float>("length");
      auto &velarr = prim->attr<vec3f>("vel");
      prim->lines.resize(prim->lines.size() + size);

      #pragma omp parallel for
      for(int i=prim->size()-size; i<prim->size(); i++)
      {
        auto p0 = pos[i-size];
        
        auto p1 = vec_to_other<openvdb::Vec3R>(p0);
        auto p2 = vecField->worldToIndex(p1);
        auto vel = openvdb::tools::BoxSampler::sample(vecField->m_grid->tree(), p2);
        velarr[i-size] = other_to_vec<3>(vel);
        auto pend = p0;
        if(lengtharr[i-size]<maxlength && maxlength>0)
        {
            pend += dt * other_to_vec<3>(vel);
        }
        pos[i] = pend;
        velarr[i] = velarr[i-size];
        lengtharr[i] = lengtharr[i-size] + length(pend - p0);
        prim->lines[i-size] = zeno::vec2i(i-size, i);
      }
    }
    

    
    
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(TraceOneStep, {
                                     {"prim", "dt", "size", "steps", "maxlength", "vecField"},
                                     {"prim"},
                                     {},
                                     {"openvdb"},
                                 });
}
