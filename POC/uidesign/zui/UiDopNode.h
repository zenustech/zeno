#pragma once


#include "UiDopSocket.h"
#include "GraphicsRectItem.h"
#include "DopNode.h"
#include "DopSocket.h"


struct UiDopGraph;


struct UiDopNode : GraphicsRectItem {
    static constexpr float DH = 40, TH = 42, FH = 24, W = 200, BW = 3;

    std::vector<UiDopInputSocket *> inputs;
    std::vector<UiDopOutputSocket *> outputs;
    std::string name;
    std::string kind;

    DopNode *bk_node = nullptr;

    void _update_backend_data() const {
        bk_node->name = name;
        bk_node->inputs.resize(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            bk_node->inputs[i].name = inputs[i]->name;
        }
        bk_node->outputs.resize(outputs.size());
        for (int i = 0; i < outputs.size(); i++) {
            bk_node->outputs[i].name = outputs[i]->name;
        }
    }

    void update_sockets() {
        for (int i = 0; i < inputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            inputs[i]->position = {UiDopSocket::R, -y};
        }
        for (int i = 0; i < outputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            outputs[i]->position = {W - UiDopSocket::R, -y};
        }
        auto h = std::max(outputs.size(), inputs.size()) * DH;
        bbox = {0, -h, W, h + TH};

        _update_backend_data();
    }

    UiDopInputSocket *add_input_socket() {
        auto p = add_child<UiDopInputSocket>();
        inputs.push_back(p);
        return p;
    }

    UiDopOutputSocket *add_output_socket() {
        auto p = add_child<UiDopOutputSocket>();
        outputs.push_back(p);
        return p;
    }

    UiDopNode() {
        selectable = true;
        draggable = true;
        bbox = {0, 0, W, TH};
    }

    UiDopGraph *get_parent() const {
        return (UiDopGraph *)(parent);
    }

    void paint() const override {
        if (selected) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.125f, 0.375f, 0.425f);
        }
        glRectf(bbox.x0 - BW, bbox.y0 - BW, bbox.x0 + bbox.nx + BW, bbox.y0 + bbox.ny + BW);

        glColor3f(0.375f, 0.375f, 0.375f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        if (selected) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.125f, 0.375f, 0.425f);
        }
        glRectf(0.f, 0.f, W, TH);

        Font font("assets/regular.ttf");
        font.set_font_size(FH);
        font.set_fixed_width(W);
        font.set_fixed_height(TH);
        glColor3f(1.f, 1.f, 1.f);
        font.render(0, FH * 0.05f, name);
    }
};
