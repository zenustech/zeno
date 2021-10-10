#pragma once


#include <zeno2/dop/Graph.h>
#include <zeno2/UI/UiDopGraph.h>
#include <rapidjson/document.h>


namespace zeno2::UI {


void deserialize(UiDopGraph *graph, rapidjson::Value const &v_graph);
rapidjson::Value serialize(UiDopGraph *graph, rapidjson::Document::AllocatorType &alloc);


}
