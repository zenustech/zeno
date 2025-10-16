#include "zeno/zeno.h"
#include "zeno/types/MaterialObject.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/ListObject.h"
#include "zeno/types/StringObject.h"
#include "zeno/types/IGeometryObject.h"
#include <zeno/types/UserData.h>
#include <tinygltf/json.hpp>
namespace zeno
{
  /*struct MakeMaterial
      : zeno::INode
  {
    virtual void apply() override
    {
      auto vert = ZImpl(get_input<zeno::StringObject>("vert"))->get();
      auto frag = ZImpl(get_input<zeno::StringObject>("frag"))->get();
      auto common = ZImpl(get_input<zeno::StringObject>("common"))->get();
      auto extensions = ZImpl(get_input<zeno::StringObject>("extensions"))->get();
      auto mtl = std::make_unique<zeno::MaterialObject>();

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
      ZImpl(set_output("mtl", std::move(mtl)));
    }
  };

  ZENDEFNODE(
      MakeMaterial,
      {
          {
              {gParamType_String, "vert", ""},
              {gParamType_String, "frag", ""},
              {gParamType_String, "common", ""},
              {gParamType_String, "extensions", ""},
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
      auto mtl = ZImpl(get_input<zeno::MaterialObject>("mtl"));
      auto s = [] (std::string const &s) { auto p = std::make_unique<StringObject>(); p->set(s); return p; };
      ZImpl(set_output("vert", s(mtl->vert)));
      ZImpl(set_output("frag", s(mtl->frag)));
      ZImpl(set_output("common", s(mtl->common)));
      ZImpl(set_output("extensions", s(mtl->extensions)));
    }
};

  struct SetMaterial
      : zeno::INode
  {
    virtual void apply() override
    {
      auto prim = ZImpl(get_input<zeno::PrimitiveObject>("prim"));
      auto mtl = safe_uniqueptr_cast<zeno::MaterialObject>(clone_input("mtl"));
      prim->mtl.reset(mtl.release());
      ZImpl(set_output("prim", std::move(prim)));
    }
  };

  ZENDEFNODE(
      SetMaterial,
      {
          {
              {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
              {gParamType_Material, "mtl", "", zeno::Socket_ReadOnly},
          },
          {
              {gParamType_Primitive, "prim"},
          },
          {},
          {
              "deprecated",
          },
      });

  struct BindMaterial
      : zeno::INode
  {
    virtual void apply() override
    {
        auto mtlid = ZImpl(get_input2<std::string>("mtlid"));

        if(ZImpl(has_input2<zeno::ListObject>("object"))) {
            auto list = ZImpl(get_input2<zeno::ListObject>("object"));
            for (auto p: list->get()) {
                p->userData()->set_string("mtlid", stdString2zs(mtlid));
            }
            set_output("object", std::move(list));
            return;
        }

      auto obj = clone_input_Geometry("object");
      
      auto mtlid2 = ZImpl(get_input2<std::string>("mtlid"));
      int matNum = obj->userData()->get_int("matNum",0);
      for(int i=0; i<matNum; i++)
      {
          auto key = "Material_" + to_string(i);
          obj->userData()->del(stdString2zs(key));
      }
      obj->userData()->set_int("matNum", 1);
      obj->userData()->set_string("Material_0", stdString2zs(mtlid2));
      obj->userData()->set_string("mtlid", stdString2zs(mtlid));
      if (obj->nfaces() > 0) {
          int matid = 0;
          obj->create_face_attr("matid", matid);
      }
      ZImpl(set_output("object", std::move(obj)));
    }
  };

  ZENDEFNODE(
      BindMaterial,
      {
          {
              {gParamType_IObject, "object", "", zeno::Socket_ReadOnly, zeno::NullControl},
              {gParamType_String, "mtlid", "Mat1"},
          },
          {
              {gParamType_IObject, "object", "", zeno::Socket_Output, zeno::NullControl},
          },
          {},
          {
              "shader",
          },
      });


    struct Material : zeno::INode
    {
        virtual void apply() override
        {
            auto obj = ZImpl(get_input<zeno::GeometryObject_Adapter>("object"));
            auto mtlid = ZImpl(get_input2<std::string>("mtlid"));
            auto mtlid2 = ZImpl(get_input2<std::string>("mtlid"));
            UserData* pUsrData = dynamic_cast<UserData*>(obj->userData());
            int matNum = pUsrData->get2<int>("matNum", 0);
            for (int i = 0; i < matNum; i++)
            {
                auto key = "Material_" + to_string(i);
                pUsrData->erase(key);
            }
            pUsrData->set2("matNum", 1);
            pUsrData->setLiterial("Material_0", std::move(mtlid2));
            pUsrData->setLiterial("mtlid", std::move(mtlid));

            int nFace = obj->nfaces();
            if (obj->nfaces() > 0) {
                obj->create_face_attr("matid", 0);
            }
            ZImpl(set_output("object", std::move(obj)));
        }
    };

    ZENDEFNODE(
        Material,
        {
            {
                {gParamType_Geometry, "object", "", zeno::Socket_ReadOnly},
                {gParamType_String, "mtlid", "Mat1"},
            },
            {
                {gParamType_Geometry, "object"},
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
            auto obj = ZImpl(get_input<zeno::IObject>("object"));
            auto isL = ZImpl(get_input2<int>("islight"));
            auto inverdir = ZImpl(get_input2<int>("invertdir"));

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

            obj->userData()->set_int("isRealTimeObject", std::move(isL));
            obj->userData()->set_int("isL", std::move(isL));
            obj->userData()->set_int("ivD", std::move(inverdir));
            ZImpl(set_output("object", std::move(obj)));
        }
    };

    ZENDEFNODE(
        BindLight,
        {
            {
                {gParamType_Primitive, "object", "", zeno::Socket_ReadOnly},
                {gParamType_Bool, "islight", "1"},// actually string or list
                {gParamType_Bool, "invertdir", "0"}
            },
            {
                {gParamType_Primitive, "object"},
            },
            {},
            {
                "shader",
            },
        });

    struct PrimAttrAsShaderBuffer : zeno::INode {

        static std::string aKey() {
            return "attrNames";
        }
        static std::string bKey() { 
            return "bindNames";
        }

        virtual void apply() override {

            auto prim = clone_input_PrimitiveObject("in");

            auto a_value = ZImpl(get_input2<std::string>(aKey(), ""));
            auto b_value = ZImpl(get_input2<std::string>(bKey(), ""));

            auto task = [](const std::string& raw){
                
                std::string segment;
                std::stringstream test(raw);
                std::vector<std::string> result;

                while(std::getline(test, segment, ','))
                {
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

            UserData* pUserData = dynamic_cast<UserData*>(prim->userData());
            pUserData->set2("ShaderAttributes", json.dump());
            set_output("out", std::move(prim));
        }
    };

    ZENDEFNODE( PrimAttrAsShaderBuffer,
    {
        {
            {gParamType_Primitive, "in"},
            {gParamType_String, PrimAttrAsShaderBuffer::aKey(), ""},
            {gParamType_String, PrimAttrAsShaderBuffer::bKey(), ""}
        },
        {
            {gParamType_Primitive, "out"},
        },
        {},
        {
            "shader",
        },
    });

} // namespace zeno

