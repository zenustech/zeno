#pragma once


#include <zs/zeno/dop/Graph.h>
#include <zs/editor/UI/UiDopGraph.h>
#include <rapidjson/document.h>


namespace zeno2::UI {


void deserialize(UiDopGraph *graph, rapidjson::Value const &v_graph);
rapidjson::Value serialize(UiDopGraph const *graph, rapidjson::Document::AllocatorType &alloc);


}
