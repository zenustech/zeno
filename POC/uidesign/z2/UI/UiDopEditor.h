#pragma once


#include <z2/UI/UiDopParam.h>
#include <z2/UI/TextEdit.h>


namespace z2::UI {


struct UiDopGraph;


struct UiDopEditor : Widget {
    TextEdit *name_edit = nullptr;
    std::vector<UiDopParam *> params;
    UiDopNode *selected = nullptr;
    UiDopGraph *graph = nullptr;

    void set_selection(UiDopNode *ptr);

    UiDopEditor();

    void clear_params();
    void update_params();
    UiDopParam *add_param();
    void paint() const override;
    void on_event(Event_Hover e) override;
};

}  // namespace z2::UI
