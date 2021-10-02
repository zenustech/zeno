#pragma once


#include "UiDopParam.h"
#include "TextEdit.h"


struct UiDopEditor : Widget {
    TextEdit *name_edit = nullptr;
    std::vector<UiDopParam *> params;
    UiDopNode *selected = nullptr;

    void set_selection(UiDopNode *ptr) {
        selected = ptr;
        clear_params();
        if (ptr) {
            for (int i = 0; i < ptr->inputs.size(); i++) {
                auto param = add_param();
                auto *socket = ptr->inputs[i];
                auto *bk_socket = &ptr->bk_node->inputs.at(i);
                param->set_bk_socket(socket, bk_socket, ptr->bk_node);
            }
        }
        update_params();
    }

    UiDopEditor() {
        bbox = {0, 0, 400, 400};
    }

    void clear_params() {
        for (auto param: params) {
            remove_child(param);
        }
        params.clear();
    }

    void update_params() {
        float y = bbox.ny - 6.f;
        for (int i = 0; i < params.size(); i++) {
            y -= params[i]->bbox.ny;
            params[i]->position = {0, y};
        }
    }

    UiDopParam *add_param() {
        auto param = add_child<UiDopParam>();
        params.push_back(param);
        return param;
    }

    void paint() const override {
        glColor3f(0.4f, 0.3f, 0.2f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};
