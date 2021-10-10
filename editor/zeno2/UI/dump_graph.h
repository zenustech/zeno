#pragma once


#include <zeno2/dop/Graph.h>
#include <zeno2/UI/UiDopGraph.h>


namespace zeno2::UI {


std::unique_ptr<dop::Graph> dump_graph(UiDopGraph *graph);


}
