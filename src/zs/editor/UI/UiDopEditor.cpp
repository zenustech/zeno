#include <zs/editor/UI/UiDopEditor.h>
#include <zs/editor/UI/UiDopGraph.h>
#include <zs/editor/UI/UiDopNode.h>


namespace zeno2::UI {


void UiDopEditor::set_selection(UiDopNode *ptr) {
    selected = ptr;
    clear_params();
    if (ptr) {
        for (int i = 0; i < ptr->inputs.size(); i++) {
            auto param = add_param();
            auto *socket = ptr->inputs[i];
            param->set_socket(socket);
        }
    }
    update_params();
}

UiDopEditor::UiDopEditor() {
    bbox = {0, 0, 400, 400};
}

void UiDopEditor::clear_params() {
    for (auto param: params) {
        remove_child(param);
    }
    params.clear();
}

void UiDopEditor::update_params() {
    float y = bbox.ny - 6.f;
    for (int i = 0; i < params.size(); i++) {
        y -= params[i]->bbox.ny;
        params[i]->position = {0, y};
    }
}

UiDopParam *UiDopEditor::add_param() {
    auto param = add_child<UiDopParam>();
    params.push_back(param);
    return param;
}

void UiDopEditor::paint() const {
    glColor3f(0.4f, 0.3f, 0.2f);
    glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
}

void UiDopEditor::on_event(Event_Hover e) {
    Widget::on_event(e);

    if (e.enter == false)
        return;

    if (graph && graph->children_selected.size()) {
        auto ptr = *graph->children_selected.begin();
        set_selection(dynamic_cast<UiDopNode *>(ptr));
    }
}


}  // namespace zeno2::UI
