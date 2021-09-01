#include <catch2/catch.hpp>
#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>

TEST_CASE("numeric operators", "[numeric]") {
    int a = 1;
    auto json = R"ZSL([["clearAllState"], ["switchGraph", "main"], ["addNode", "NumericInt", "549d5330-NumericInt"], ["setNodeParam", "549d5330-NumericInt", "value", 2], ["completeNode", "549d5330-NumericInt"], ["addNode", "SubInput", "a9d25a1f-SubInput"], ["setNodeParam", "a9d25a1f-SubInput", "type", ""], ["setNodeParam", "a9d25a1f-SubInput", "name", "input"], ["setNodeParam", "a9d25a1f-SubInput", "defl", ""], ["completeNode", "a9d25a1f-SubInput"], ["addNode", "NumericOperator", "b4327a00-NumericOperator"], ["bindNodeInput", "b4327a00-NumericOperator", "lhs", "549d5330-NumericInt", "value"], ["bindNodeInput", "b4327a00-NumericOperator", "rhs", "a9d25a1f-SubInput", "port"], ["setNodeParam", "b4327a00-NumericOperator", "op_type", "add"], ["completeNode", "b4327a00-NumericOperator"], ["addNode", "SubOutput", "cbbfef17-SubOutput"], ["bindNodeInput", "cbbfef17-SubOutput", "port", "b4327a00-NumericOperator", "ret"], ["setNodeParam", "cbbfef17-SubOutput", "type", ""], ["setNodeParam", "cbbfef17-SubOutput", "name", "output"], ["setNodeParam", "cbbfef17-SubOutput", "defl", ""], ["completeNode", "cbbfef17-SubOutput"]])ZSL";
    auto scene = zeno::createScene();
    scene->loadScene(json);
    scene->switchGraph("main");
    auto input = std::make_shared<zeno::NumericObject>(40);
    scene->getGraph().setGraphInput("input", input);
    scene->getGraph().applyGraph();
    auto output = scene->getGraph().getGraphOutput<zeno::NumericObject>("output");
    REQUIRE(output->get<int>() == 42);
}
