#pragma once


#include <z2/UI/UiDopGraph.h>
#include <z2/UI/UiDopEditor.h>


namespace z2::UI {


struct UiDopScene : Widget {
    UiDopGraph *graph;
    UiDopEditor *editor;

    UiDopScene() {
        graph = add_child<UiDopGraph>();
        graph->bbox = {0, 0, 1088, 440};
        graph->position = {0, 0};

        editor = add_child<UiDopEditor>();
        editor->bbox = {0, 0, 512, 440};
        editor->position = {1088, 0};

        graph->editor = editor;
        editor->graph = graph;
        bbox = {0, 0, 1024, 768};
    }
};


}
