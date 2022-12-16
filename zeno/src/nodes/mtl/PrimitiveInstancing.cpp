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
    /*
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
    */

    struct MakeInstMatrix
        : zeno::INode
    {
        virtual void apply() override
        {
            auto instMat = std::make_shared<zeno::InstMatrixObject>();
            if (has_input("translate"))
            {
                instMat->translate = get_input<zeno::NumericObject>("translate")->get<zeno::vec3f>();
            }
            if (has_input("rotate"))
            {
                instMat->rotate = get_input<zeno::NumericObject>("rotate")->get<zeno::vec3f>();
            }
            if (has_input("scale"))
            {
                instMat->scale = get_input<zeno::NumericObject>("scale")->get<zeno::vec3f>();
            }
            set_output("instMat", instMat);                    
        };
    };

    ZENDEFNODE(
        MakeInstMatrix,
        {
            {
                {"vec3f", "translate", "0,0,0"},
                {"vec3f", "rotate", "0,0,0"},
                {"vec3f", "scale", "1,1,1"},
            },
            {
                {"instMat"},
            },
            {},
            {
                "shader",
            },
        }
    );

    struct MakeInstancing
        : zeno::INode
    {
        virtual void apply() override
        {
            auto prim = std::make_shared<zeno::PrimitiveObject>();
            auto isInst = get_input2<int>("isInst");
            auto instID = get_input2<std::string>("instID");
            prim->userData().set2("isInst", std::move(isInst));
            prim->userData().setLiterial("instID", std::move(instID));

            if (has_input("instMatList"))
            {
                auto instMatList = get_input<zeno::ListObject>("instMatList")->get<zeno::InstMatrixObject>();
                auto numInstMatList = instMatList.size();
                prim->verts.resize(numInstMatList);

                auto &tranlate = prim->add_attr<zeno::vec3f>("pos");
                auto &rotate = prim->add_attr<zeno::vec3f>("nrm");
                auto &scale = prim->add_attr<zeno::vec3f>("cls");

                for (std::size_t i = 0; i < numInstMatList; ++i)
                {
                    auto& instMat = instMatList[i];
                    tranlate[i] = instMat->translate;
                    rotate[i] = instMat->rotate;
                    scale[i] = instMat->scale;
                }
            }

            set_output("object", std::move(prim));
        }
    };

    ZENDEFNODE(
        MakeInstancing,
        {
            {
                {"bool", "isInst", "1"},
                {"string", "instID", "Inst1"},
                {"list", "instMatList"},
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

    struct BindInstancing
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
        BindInstancing,
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
