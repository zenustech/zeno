#pragma once


#include <zeno/UI/UiDopSocket.h>
#include <zeno/UI/GraphicsRectItem.h>
#include <zeno/dop/DopNode.h>
#include <zeno/dop/DopSocket.h>


namespace zeno::UI {


struct UiDopGraph;


struct UiDopNode : GraphicsRectItem {
    static constexpr float DH = 40, TH = 42, FH = 24, W = 200, BW = 3;

    std::vector<UiDopInputSocket *> inputs;
    std::vector<UiDopOutputSocket *> outputs;
    std::string name;
    std::string kind;

    DopNode *bk_node = nullptr;

    UiDopNode();

    void update_sockets();
    void _update_backend_data() const;
    UiDopInputSocket *add_input_socket();
    UiDopOutputSocket *add_output_socket();
    UiDopGraph *get_parent() const;
    void paint() const override;
};


}  // namespace zeno::UI
