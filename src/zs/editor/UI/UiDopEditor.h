#pragma once


#include <zs/editor/UI/UiDopParam.h>
#include <zs/editor/UI/TextEdit.h>


namespace zs::editor::UI {


struct UiDopGraph;


struct UiDopEditor : Widget {
    TextEdit *name_edit = nullptr;
    std::vector<UiDopParam *> params;
    UiDopNode *selected = nullptr;
    UiDopGraph *graph = nullptr;

    UiDopEditor();
    void clear_params();
    void update_params();
    UiDopParam *add_param();
    void paint() const override;
    void on_event(Event_Hover e) override;
    void set_selection(UiDopNode *ptr);
};


}  // namespace zs::editor::UI
