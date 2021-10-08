#pragma once


#include <zeno2/UI/UiDopSocket.h>
#include <zeno2/UI/GraphicsRectItem.h>


namespace zeno2::UI {


struct UiDopGraph;


struct UiDopNode : GraphicsRectItem {
    static constexpr float DH = 40, TH = 42, FH = 24, W = 200, BW = 3;

    std::vector<UiDopInputSocket *> inputs;
    std::vector<UiDopOutputSocket *> outputs;
    std::string name;
    std::string kind;
    bool failed = false;

    UiDopNode();
    void update_sockets();
    UiDopInputSocket *add_input_socket();
    UiDopOutputSocket *add_output_socket();
    UiDopGraph *get_parent() const;
    void paint() const override;
};


}  // namespace zeno2::UI
