#include "zeno/zeno.h"
#include "zeno/types/MaterialObject.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/StringObject.h"
#include <zeno/types/UserData.h>
#include <zeno/types/AlternativeAttribute.h>
#include <zeno/types/PrimitiveUV.h>
#include <type_traits>
#include <zeno/funcs/PrimitiveUtils.h>
#include <cmath>
#include <tinygltf/json.hpp>
namespace zeno
{
  /*struct MakeMaterial
      : zeno::INode
  {
    virtual void apply() override
    {
      auto vert = get_input<zeno::StringObject>("vert")->get();
      auto frag = get_input<zeno::StringObject>("frag")->get();
      auto common = get_input<zeno::StringObject>("common")->get();
      auto extensions = get_input<zeno::StringObject>("extensions")->get();
      auto mtl = std::make_shared<zeno::MaterialObject>();

      if (vert.empty()) vert = R"(
#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;

attribute vec3 vPosition;
attribute vec3 vColor;
attribute vec3 vNormal;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;

void main()
{
  position = vPosition;
  iColor = vColor;
  iNormal = vNormal;

  gl_Position = mVP * vec4(position, 1.0);
}
)";

      if (frag.empty()) frag = R"(
#version 120

uniform mat4 mVP;
uniform mat4 mInvVP;
uniform mat4 mView;
uniform mat4 mProj;
uniform mat4 mInvView;
uniform mat4 mInvProj;
uniform bool mSmoothShading;
uniform bool mRenderWireframe;

varying vec3 position;
varying vec3 iColor;
varying vec3 iNormal;

void main()
{
  gl_FragColor = vec4(8.0, 0.0, 0.0, 1.0);
}
)";

      mtl->vert = vert;
      mtl->frag = frag;
      mtl->common = common;
      mtl->extensions = extensions;
      set_output("mtl", std::move(mtl));
    }
  };

  ZENDEFNODE(
      MakeMaterial,
      {
          {
              {"string", "vert", ""},
              {"string", "frag", ""},
              {"string", "common", ""},
              {"string", "extensions", ""},
          },
          {
              {"material", "mtl"},
          },
          {},
          {
              "shader",
          },
      });*/

struct ExtractMaterialShader : zeno::INode
{
    virtual void apply() override {
      auto mtl = get_input<zeno::MaterialObject>("mtl");
      auto s = [] (std::string const &s) { auto p = std::make_shared<StringObject>(); p->set(s); return p; };
      set_output("vert", s(mtl->vert));
      set_output("frag", s(mtl->frag));
      set_output("common", s(mtl->common));
      set_output("extensions", s(mtl->extensions));
    }
};

  struct SetMaterial
      : zeno::INode
  {
    virtual void apply() override
    {
      auto prim = get_input<zeno::PrimitiveObject>("prim");
      auto mtl = get_input<zeno::MaterialObject>("mtl");
      prim->mtl = mtl;
      set_output("prim", std::move(prim));
    }
  };

  ZENDEFNODE(
      SetMaterial,
      {
          {
              {"primitive", "prim"},
              {"material", "mtl"},
          },
          {
              {"primitive", "prim"},
          },
          {},
          {
              "deprecated",
          },
      });

  struct BindMaterial : zeno::INode
  {
    virtual void apply() override
    {
        auto mtlid = get_input2<std::string>("mtlid");

        if( has_input2<zeno::ListObject>("object") ) {
            auto list = get_input2<zeno::ListObject>("object");
            for (auto p: list->arr) {
                auto np = std::dynamic_pointer_cast<PrimitiveObject>(p);
                np->userData().setLiterial("mtlid", std::move(mtlid));
            }
            set_output("object", std::move(list));
            return;
        }

      auto obj = get_input<zeno::PrimitiveObject>("object");
      
      auto mtlid2 = get_input2<std::string>("mtlid");
      int matNum = obj->userData().get2<int>("matNum",0);
      for(int i=0; i<matNum; i++)
      {
          auto key = "Material_" + to_string(i);
          obj->userData().erase(key);
      }
      obj->userData().set2("matNum", 1);
      obj->userData().setLiterial("Material_0", std::move(mtlid2));
      obj->userData().setLiterial("mtlid", std::move(mtlid));
      if(obj->tris.size()>0)
      {
          obj->tris.add_attr<int>("matid");
          obj->tris.attr<int>("matid").assign(obj->tris.size(),0);
      }
      if(obj->quads.size()>0)
      {
          obj->quads.add_attr<int>("matid");
          obj->quads.attr<int>("matid").assign(obj->quads.size(),0);
      }
      if(obj->polys.size()>0)
      {
          obj->polys.add_attr<int>("matid");
          obj->polys.attr<int>("matid").assign(obj->polys.size(),0);
      }
      set_output("object", std::move(obj));
    }
  };

  ZENDEFNODE(
      BindMaterial,
      {
          {
              {"object"},
              {"string", "mtlid", "Mat1"},
          },
          {
              {"object"},
          },
          {},
          {
              "shader",
          },
      });

    struct BindLight
        : zeno::INode
    {
        virtual void apply() override
        {
            auto obj = get_input<zeno::IObject>("object");
            auto isL = get_input2<int>("islight");
            auto inverdir = get_input2<int>("invertdir");

            auto prim = dynamic_cast<zeno::PrimitiveObject *>(obj.get());

            zeno::vec3f clr;
            if (! prim->verts.has_attr("clr")) {
                auto &clr = prim->verts.add_attr<zeno::vec3f>("clr");
                clr[0] = zeno::vec3f(30000.0f, 30000.0f, 30000.0f);
            }

            if(inverdir){
                for(int i=0;i<prim->tris.size(); i++){
                    int tmp = prim->tris[i][2];
                    prim->tris[i][2] = prim->tris[i][0];
                    prim->tris[i][0] = tmp;
                }
            }

            obj->userData().set2("isRealTimeObject", std::move(isL));
            obj->userData().set2("isL", std::move(isL));
            obj->userData().set2("ivD", std::move(inverdir));
            set_output("object", std::move(obj));
        }
    };

    ZENDEFNODE(
        BindLight,
        {
            {
                {"object"},
                {"bool", "islight", "1"},// actually string or list
                {"bool", "invertdir", "0"}
            },
            {
                {"object"},
            },
            {},
            {
                "shader",
            },
        });

    // Prepare several mesh-local bindings in one operation. The node snapshots
    // values and triangulates its own copy; the renderer's primary UV topology
    // and all original attribute buffers retain their existing implementation.
    struct PrepareAlternativeAttribute : zeno::INode {
        void apply() override {
            const auto input = get_input<PrimitiveObject>("prim");
            const auto rows = nlohmann::json::parse(get_input2<std::string>("bindings"));
            if (!rows.is_array()) throw std::runtime_error("Alternative bindings must be an array");
            auto prim = std::make_shared<PrimitiveObject>(*input);
            auto metadata = nlohmann::json::array();
            if (input->userData().has("AlternativeAttributes")) {
                auto previous = nlohmann::json::parse(input->userData().get2<std::string>("AlternativeAttributes"));
                if (previous.at("topology") != alternativeTopologyKey(*input))
                    throw std::runtime_error("Alternative attributes require preparation after topology edits");
                metadata = previous.at("bindings");
                for (const auto &binding : metadata) {
                    const auto alter=binding.at("alter").get<int>();
                    if (alter<1 || unsigned(alter)>MeshAttributeAlterCount)
                        throw std::runtime_error("Prepared alternative Alter must be between 1 and 8: " +
                            binding.at("attr").get<std::string>() + "/" + std::to_string(alter));
                }
            }
            // Match the existing quad/polygon triangulators' ordering. This
            // remap is local to preparation; no new primitive domain is added.
            struct SourceTriangle { vec3i points, corners; int face, kind; };
            std::vector<SourceTriangle> remap;
            for (int f=0; f<input->tris.size(); ++f)
                remap.push_back({input->tris[f], {0,1,2}, f, 0});
            for (int f=0; f<input->quads.size(); ++f) {
                const auto q=input->quads[f];
                remap.push_back({{q[0],q[1],q[2]}, {0,1,2}, f, 1});
                remap.push_back({{q[0],q[2],q[3]}, {0,2,3}, f, 1});
            }
            for (int f=0; f<input->polys.size(); ++f) {
                const auto p=input->polys[f];
                if (p[0]<0 || p[1]<0 || size_t(p[0])+p[1]>input->loops.size())
                    throw std::runtime_error("Invalid polygon range in alternative attribute source");
                for (int k=2; k<p[1]; ++k) {
                    vec3i c(p[0],p[0]+k-1,p[0]+k);
                    remap.push_back({{input->loops[c[0]],input->loops[c[1]],input->loops[c[2]]}, c, f, 2});
                }
            }
            for (const auto &t : remap) for (int k=0; k<3; ++k)
                if (t.points[k]<0 || size_t(t.points[k])>=input->verts.size())
                    throw std::runtime_error("Alternative attribute mesh point index out of range");
            primTriangulateQuads(prim.get());
            primTriangulate(prim.get());
            if (prim->tris.size()!=remap.size()) throw std::runtime_error("Alternative triangulation mismatch");
            for (size_t t=0; t<remap.size(); ++t)
                if (anytrue(prim->tris[t]!=remap[t].points)) throw std::runtime_error("Alternative corner remap mismatch");
            auto readUV = [](const auto &container, const std::string &name, size_t index) -> vec2f {
                vec2f value;
                if (container.template attr_is<vec2f>(name)) {
                    const auto &values=container.template attr<vec2f>(name);
                    if (values.size()!=container.size()) throw std::runtime_error("Wrong alternative source count: " + name);
                    value=values.at(index);
                }
                else if (container.template attr_is<vec3f>(name)) {
                    const auto &values=container.template attr<vec3f>(name);
                    if (values.size()!=container.size()) throw std::runtime_error("Wrong alternative source count: " + name);
                    auto v=values.at(index); value={v[0],v[1]};
                } else throw std::runtime_error("Alternative UV source must be vec2f or vec3f: " + name);
                if (!std::isfinite(value[0]) || !std::isfinite(value[1]))
                    throw std::runtime_error("Non-finite alternative UV source: " + name);
                return value;
            };
            auto readVector = [](const auto &container, const std::string &name, size_t index) -> vec3f {
                if (!container.template attr_is<vec3f>(name))
                    throw std::runtime_error("Alternative normal/color/tangent source must be vec3f: " + name);
                const auto &values=container.template attr<vec3f>(name);
                if (values.size()!=container.size()) throw std::runtime_error("Wrong alternative source count: " + name);
                const auto value=values.at(index);
                for (int k=0;k<3;++k) if (!std::isfinite(value[k]))
                    throw std::runtime_error("Non-finite alternative vector source: " + name);
                return value;
            };
            // Domain storage is independent of the adapter's value type.
            auto storeAttribute=[&](const std::string &name,const std::string &domain,auto fetch) {
                using T=decltype(fetch(size_t(0),0));
                if(domain=="point") {
                    auto &values=prim->verts.add_attr<T>(name);
                    for(size_t i=0;i<values.size();++i)values[i]=fetch(i,0);
                } else if(domain=="face") {
                    auto &values=prim->tris.add_attr<T>(name);
                    for(size_t i=0;i<values.size();++i)values[i]=fetch(i,0);
                } else if(domain=="corner") {
                    for(int k=0;k<3;++k) {
                        auto &values=prim->tris.add_attr<T>(name+"_corner_"+std::to_string(k));
                        for(size_t i=0;i<values.size();++i)values[i]=fetch(i,k);
                    }
                } else throw std::runtime_error("Unknown alternative source domain: "+domain);
            };
            const auto imported = primitiveUVSets(*input);
            for (const auto &row : rows) {
                const auto source=row.at("source").get<std::string>();
                const auto domain=row.at("domain").get<std::string>();
                const auto attr=row.at("attr").get<std::string>();
                const auto index=row.at("alter").get<int>();
                if (source.empty() || source.rfind("__alternative_",0)==0)
                    throw std::runtime_error("Invalid alternative source attribute name");
                const auto semantic=meshAttributeSemantic(attr);
                if (index<1 || unsigned(index)>MeshAttributeAlterCount)
                    throw std::runtime_error("Alternative Alter must be between 1 and 8: " + attr + "/" + std::to_string(index));
                for (const auto &existing : metadata)
                    if (existing.at("alter")==index && existing.at("attr")==attr)
                        throw std::runtime_error("Duplicate alternative binding: " + attr + "/" + std::to_string(index));
                const auto name=alternativeAttributeName(attr,index);
                const PrimitiveUVSet *uvSet = nullptr;
                if (semantic == MeshAttributeSemantic::UV) {
                    for (const auto &set : imported) {
                        if (set.name == source || set.source == source ||
                            std::find(set.aliases.begin(), set.aliases.end(), source) != set.aliases.end()) {
                            if (uvSet) {
                                throw std::runtime_error("Ambiguous source UV set name: " + source);
                            }
                            uvSet = &set;
                        }
                    }
                }
                if (uvSet) {
                    storeAttribute(name, domain, [&](size_t t, int k) -> vec2f {
                        vec2f value{};
                        bool valid = false;
                        if (domain == "point") {
                            valid =
                                uvSet->domain == "point" && primitiveUVPointValue(*input, *uvSet, t, value);
                        } else if (domain == "corner") {
                            const auto &r = remap[t];
                            if (r.kind == 0) {
                                valid = primitiveUVValue(*input, *uvSet, r.face, k, value);
                            } else if (r.kind == 2) {
                                valid = primitiveUVLoopValue(*input, *uvSet, r.corners[k], value);
                            }
                        }
                        if (!valid) {
                            throw std::runtime_error("Invalid source UV set or domain: " + source + "/" +
                                                     domain);
                        }
                        return value;
                    });
                    metadata.push_back(row);
                    continue;
                }
                auto prepareValues=[&](auto read) {
                    using Value=decltype(read(input->verts,source,size_t(0)));
                    storeAttribute(name,domain,[&](size_t t,int k)->Value {
                        if(domain=="point") return read(input->verts,source,t);
                        const auto &r=remap[t];
                        if(domain=="face")
                            return r.kind==0?read(input->tris,source,r.face):r.kind==1?read(input->quads,source,r.face):read(input->polys,source,r.face);
                        if (r.kind == 0) {
                            if constexpr (std::is_same_v<Value, vec2f>) {
                                int set = primitiveUVIndex(source);
                                if (source.rfind("auv_", 0) == 0) {
                                    set = primitiveUVIndex("uvs_" + source.substr(4));
                                }
                                if (set >= 0) {
                                    vec2f value;
                                    if (!primitiveUVTriangleValue(*input, set, r.face, k, value)) {
                                        throw std::runtime_error("Invalid triangulated UV source: " + source);
                                    }
                                    return value;
                                }
                            }
                            const auto name = primitiveCornerName(source, k);
                            return read(input->tris,
                                        input->tris.has_attr(name) ? name : source + std::to_string(k),
                                        r.face);
                        }
                        if(r.kind==1) return read(input->quads,source+std::to_string(r.corners[k]),r.face);
                        if constexpr(std::is_same_v<Value,vec2f>) {
                            // Only UV sources may refer to an imported UV palette.
                            if(input->loops.attr_is<int>(source)) {
                                const auto index=input->loops.attr<int>(source).at(r.corners[k]);
                                if(index<0) throw std::runtime_error("Negative alternative UV palette index");
                                const auto set = primitiveUVIndex(source);
                                const auto palette = set >= 0 ? primitiveUVPalette(*input, set) : nullptr;
                                if (!palette) {
                                    throw std::runtime_error("Unknown alternative UV palette: " + source);
                                }
                                const auto value = palette->values.at(index);
                                if(!std::isfinite(value[0]) || !std::isfinite(value[1])) throw std::runtime_error("Non-finite alternative UV: "+source);
                                return value;
                            }
                        }
                        return read(input->loops,source,r.corners[k]);
                    });
                };
                if(semantic==MeshAttributeSemantic::UV) 
                    prepareValues(readUV);
                else 
                    prepareValues(readVector);
                metadata.push_back(row);
            }
            nlohmann::json info={{"bindings",metadata},{"topology",alternativeTopologyKey(*prim)}};
            prim->userData().set2("AlternativeAttributes",info.dump());
            set_output("prim",std::move(prim));
        }
    };
    ZENDEFNODE(PrepareAlternativeAttribute, {
        {{"primitive","prim"},{"string","bindings",R"([{"source":"uv","domain":"point","attr":"uv","alter":1}])"}},
        {{"primitive","prim"}}, {}, {"shader"}
    });

    struct PrimAttrAsShaderBuffer : zeno::INode {

        static std::string aKey() {
            return "attrNames";
        }
        static std::string bKey() { 
            return "bindNames";
        }

        virtual void apply() override {

            auto prim = get_input2<zeno::PrimitiveObject>("in");

            auto a_value = get_input2<std::string>(aKey(), "");
            auto b_value = get_input2<std::string>(bKey(), "");

            auto task = [](const std::string& raw){
                
                std::string segment;
                std::stringstream test(raw);
                std::vector<std::string> result;

                while(std::getline(test, segment, ','))
                {
                    segment.erase(std::remove(segment.begin(), segment.end(), ' '), segment.end());
                    result.push_back(segment);
                }
                return result;
            };

            std::vector<std::string> a_list = task(a_value);
            std::vector<std::string> b_list = task(b_value);

            if (a_list.size() != b_list.size()) {
                throw std::runtime_error("buffer count doesn't match attr count");
            }

            nlohmann::json json;

            for (size_t i=0; i<a_list.size(); ++i) {
                auto a = a_list[i];
                auto b = b_list[i];
                json[a] = b;
            }

            prim->userData().set2("ShaderAttributes", json.dump());
            set_output("out", std::move(prim));
        }
    };

    ZENDEFNODE( PrimAttrAsShaderBuffer,
    {
        {
            {"in"},
            {"string", PrimAttrAsShaderBuffer::aKey(), ""},
            {"string", PrimAttrAsShaderBuffer::bKey(), ""}
        },
        {
            {"out"},
        },
        {},
        {
            "shader",
        },
    });

} // namespace zeno
