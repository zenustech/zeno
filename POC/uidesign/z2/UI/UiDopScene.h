#pragma once


#include <z2/UI/UiDopGraph.h>
#include <z2/UI/UiDopEditor.h>


namespace z2::UI {


struct UiDopScene : Widget {
    UiDopGraph *graph;
    UiDopEditor *editor;
    std::any view_result;
    SignalSlot on_result_changed;

    void set_view_result(std::any val) {
        view_result = val;
        on_result_changed();
    }

    UiDopScene() {
        graph = add_child<UiDopGraph>();
        graph->bbox = {0, 0, 1088, 440};
        graph->position = {0, 0};

        editor = add_child<UiDopEditor>();
        editor->bbox = {0, 0, 512, 440};
        editor->position = {1088, 0};

        graph->editor = editor;
        editor->graph = graph;
        bbox = {0, 0, 1600, 440};
    }
};


}
