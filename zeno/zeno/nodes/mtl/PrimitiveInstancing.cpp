#include <zeno/zeno.h>
#include <zeno/types/InstancingObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
// #include <zeno/types/MatrixObject.h>
#include <zeno/types/PrimitiveObject.h>
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

            std::size_t modelMatricesIndex = 0;
            if (has_input("modelMatrices"))
            {
                auto modelMatrices = get_input<zeno::ListObject>("modelMatrices")->get<std::shared_ptr<zeno::MatrixObject>>();
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

            set_output("inst", std::move(inst));
        }

    }; // struct MakeInstancing

    ZENDEFNODE(
        MakeInstancing,
        {
            {
                {"int", "amount", "1"},
                {"list", "modelMatrices"},
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

} // namespace zeno
