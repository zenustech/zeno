#include <zeno/zeno.h>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/MatrixObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cmath>

namespace zeno
{
    struct MakeInstancing
        : zeno::INode
    {
        virtual void apply() override
        {
            auto inst = std::make_shared<zeno::InstancingObject>();

            auto amount = get_input<zeno::NumericObject>("amount")->get<int>();
            inst->amount = amount;
            inst->modelMatrices.reserve(amount);
            inst->timeList.reserve(amount);

            std::size_t modelMatricesIndex = 0;
            if (has_input("modelMatrices"))
            {
                auto modelMatrices = get_input<zeno::ListObject>("modelMatrices")->get<zeno::MatrixObject>();
                auto modelMatricesSize = modelMatrices.size();
                auto firstLoopCnt = std::min(static_cast<std::size_t>(amount), modelMatricesSize);
                for (; modelMatricesIndex < firstLoopCnt; ++modelMatricesIndex)
                {
                    const auto &modelMatrix = std::get<glm::mat4>(modelMatrices[modelMatricesIndex]->m);
                    inst->modelMatrices.push_back(modelMatrix);
                }
            }
            for (; modelMatricesIndex < amount; ++modelMatricesIndex)
            {
                inst->modelMatrices.push_back(glm::mat4(1.0f));
            }

            std::size_t timeListIndex = 0;
            if (has_input("timeList"))
            {
                auto timeList = get_input<zeno::ListObject>("timeList")->get<zeno::NumericObject>();
                auto timeListSize = timeList.size();
                auto firstLoopCnt = std::min(static_cast<std::size_t>(amount), timeListSize);
                for (; timeListIndex < firstLoopCnt; ++timeListIndex)
                {
                    const auto &time = timeList[timeListIndex]->get<float>();
                    inst->timeList.push_back(time);
                }
            }
            for (; timeListIndex < amount; ++timeListIndex)
            {
                inst->timeList.push_back(0.0f);
            }

            auto deltaTime = get_input<zeno::NumericObject>("deltaTime")->get<float>();
            inst->deltaTime = deltaTime;

            if (has_input("framePrims"))
            {
                auto framePrims = get_input<zeno::ListObject>("framePrims")->get<zeno::PrimitiveObject>();
                auto frameAmount = framePrims.size();  
                auto &vertexFrameBuffer = inst->vertexFrameBuffer;
                vertexFrameBuffer.resize(frameAmount);

                std::size_t vertexAmount = 0;
                if (frameAmount > 0)
                {
                    const auto &pos = framePrims[0]->attr<zeno::vec3f>("pos");
                    vertexAmount = pos.size();
                }

                for (int i = 0; i < frameAmount; ++i)
                {
                    const auto &pos = framePrims[i]->attr<zeno::vec3f>("pos");
                    if (vertexAmount != pos.size())
                    {
                        throw zeno::Exception("vertex amount is not a same!");
                    }
                    vertexFrameBuffer[i] = pos;
                } 
            }

            set_output("inst", std::move(inst));
        }

    }; // struct MakeInstancing

    ZENDEFNODE(
        MakeInstancing,
        {
            {
                {"int", "amount", "1"},
                {"list", "modelMatrices"},
                {"float", "deltaTime", "0.0"},
                {"list", "timeList"},
                {"list", "framePrims"},
            },
            {
                {"instancing", "inst"},
            },
            {},
            {
                "shader",
            },
        });

    struct SetInstancing
        : zeno::INode
    {
        virtual void apply() override
        {
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            auto inst = get_input<zeno::InstancingObject>("inst");
            prim->inst = inst;
            set_output("prim", std::move(prim));
        }

    }; // struct SetInstancing

    ZENDEFNODE(
        SetInstancing,
        {
            {
                {"primitive", "prim"},
                {"instancing", "inst"},
            },
            {
                {"primitive", "prim"},
            },
            {},
            {
                "shader",
            },
        });

    struct BecomeRtInst
        : zeno::INode
    {
        virtual void apply() override
        {
            auto obj = get_input<zeno::IObject>("object");
            auto isInst = get_input2<int>("isInst");
            auto instID = get_input2<std::string>("instID");
            auto onbType = get_input2<std::string>("onbType");

            obj->userData().set2("isInst", std::move(isInst));
            obj->userData().setLiterial("instID", std::move(instID));
            obj->userData().setLiterial("onbType", std::move(onbType));

            /* test
            auto prim = dynamic_cast<zeno::PrimitiveObject *>(obj.get());
            prim->verts.resize(3);
            auto &pos = prim->add_attr<zeno::vec3f>("pos");
            auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
            auto &clr = prim->add_attr<zeno::vec3f>("clr");
            auto &uv = prim->add_attr<zeno::vec3f>("uv");
            auto &tang = prim->add_attr<zeno::vec3f>("tang");

            pos = {{2, 0, 0}, {0, 0, 0}, {-2, 0, 0}}; // translation
            nrm = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // direction
            clr = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}; // scaling
            uv = {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}};
            tang = {{0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
            */

            set_output("object", std::move(obj));
        }
    };

    ZENDEFNODE(
        BecomeRtInst,
        {
            {
                {"object"},
                {"bool", "isInst", "1"},
                {"string", "instID", "Inst1"},
                {"enum XYZ YXZ YZX ZYX ZXY XZY Matrix None", "onbType", "XYZ"},
            },
            {
                {"object"},
            },
            {},
            {
                "shader",
            },
        }
    );

    struct BindRtInst
        : zeno::INode
    {
        virtual void apply() override
        {
            auto obj = get_input<zeno::IObject>("object");
            auto instID = get_input2<std::string>("instID");

            obj->userData().setLiterial("instID", std::move(instID));

            set_output("object", std::move(obj));
        }
    };

    ZENDEFNODE(
        BindRtInst,
        {
            {
                {"object"},
                {"string", "instID", "Inst1"},
            },
            {
                {"object"},
            },
            {},
            {
                "shader",
            },
        }
    );

} // namespace zeno
