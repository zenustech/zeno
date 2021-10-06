#pragma once


#include <z2/UI/UiDopSocket.h>
#include <z2/UI/GraphicsRectItem.h>
#include <z2/dop/dop.h>


namespace z2::UI {


struct UiDopGraph;


struct UiDopNode : GraphicsRectItem {
    static constexpr float DH = 40, TH = 42, FH = 24, W = 200, BW = 3;

    std::vector<UiDopInputSocket *> inputs;
    std::vector<UiDopOutputSocket *> outputs;
    std::string name;
    std::string kind;

    dop::Node *bk_node = nullptr;

    UiDopNode();
    void update_sockets();
    void _update_backend_data() const;
    UiDopInputSocket *add_input_socket();
    UiDopOutputSocket *add_output_socket();
    void set_position(Point pos) override;
    UiDopGraph *get_parent() const;
    void paint() const override;
};


}  // namespace z2::UI
