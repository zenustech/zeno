#pragma once


#include <zeno2/UI/UiDopParam.h>
#include <zeno2/UI/TextEdit.h>


namespace zeno2::UI {


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


}  // namespace zeno2::UI
