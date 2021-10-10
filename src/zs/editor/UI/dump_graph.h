#pragma once


#include <zs/zeno/dop/Graph.h>
#include <zs/editor/UI/UiDopGraph.h>


namespace zeno2::UI {


std::unique_ptr<dop::Graph> dump_graph(UiDopGraph const *graph);


}
