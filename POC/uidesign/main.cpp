#include "zui/stdafx.h"
#include "zed/stdafx.h"

// END node data structures

// BEG node editor ui


struct GraphicsLineItem : GraphicsWidget {
    static constexpr float LW = 5.f;

    virtual Point get_from_position() const = 0;
    virtual Point get_to_position() const = 0;

    bool contains_point(Point p) const override {
        auto p0 = get_from_position();
        auto p1 = get_to_position();
        auto a = p - p0;
        auto b = p1 - p0;
        auto l = std::sqrt(b.x * b.x + b.y * b.y);
        b = b * (1.f / l);
        auto c = a.x * b.y - a.y * b.x;
        auto d = a.x * b.x + a.y * b.y;
        if (std::max(d, l - d) - l > LW)
            return false;
        return std::abs(c) < LW;  // actually LW*2 collision width
    }

    virtual Color get_line_color() const {
        if (selected || hovered) {
            return {0.75f, 0.5f, 0.375f};
        } else {
            return {0.375f, 0.5f, 1.0f};
        }
    }

    void paint() const override {
        glColor3fv(get_line_color().data());
        auto [sx, sy] = get_from_position();
        auto [dx, dy] = get_to_position();
        glLineWidth(LW);
        glBegin(GL_LINE_STRIP);
        glVertex2f(sx, sy);
        glVertex2f(dx, dy);
        glEnd();
    }
};


struct UiDopLink : GraphicsLineItem {
    UiDopOutputSocket *from_socket;
    UiDopInputSocket *to_socket;

    UiDopLink(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket)
        : from_socket(from_socket), to_socket(to_socket)
    {
        from_socket->attach_link(this);
        to_socket->attach_link(this);
        selectable = true;
        zvalue = 1.f;
    }

    Point get_from_position() const override {
        return from_socket->position + from_socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return to_socket->position + to_socket->get_parent()->position;
    }

    UiDopGraph *get_parent() const {
        return (UiDopGraph *)(parent);
    }
};


struct UiDopPendingLink : GraphicsLineItem {
    UiDopSocket *socket;

    UiDopPendingLink(UiDopSocket *socket)
        : socket(socket)
    {
        zvalue = 3.f;
    }

    Color get_line_color() const override {
        return {0.75f, 0.5f, 0.375f};
    }

    Point get_from_position() const override {
        return socket->position + socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return {cur.x, cur.y};
    }

    UiDopGraph *get_parent() const {
        return (UiDopGraph *)(parent);
    }

    Widget *item_at(Point p) const override {
        return nullptr;
    }
};


struct UiDopContextMenu : Widget {
    static constexpr float EH = 32.f, EW = 210.f, FH = 20.f;

    std::vector<Button *> entries;
    std::string selection;

    SignalSlot on_selected;

    UiDopContextMenu() {
        position = {cur.x, cur.y};
        zvalue = 10.f;
    }

    Button *add_entry(std::string name) {
        auto btn = add_child<Button>();
        btn->text = name;
        btn->bbox = {0, 0, EW, EH};
        btn->font_size = FH;
        btn->on_clicked.connect([=, this] {
            selection = name;
            on_selected();
        });
        entries.push_back(btn);
        return btn;
    }

    void update_entries() {
        for (int i = 0; i < entries.size(); i++) {
            entries[i]->position = {0, -(i + 1) * EH};
        }
        bbox = {0, entries.size() * -EH, EW, entries.size() * EH};
    }
};


struct UiDopEditor;

struct UiDopGraph : GraphicsView {
    std::set<UiDopNode *> nodes;
    std::set<UiDopLink *> links;
    UiDopPendingLink *pending_link = nullptr;

    std::unique_ptr<DopGraph> bk_graph = std::make_unique<DopGraph>();

    // must invoke these two functions rather than operate on |links| and
    // |remove_child| directly to prevent bad pointer
    bool remove_link(UiDopLink *link) {
        if (remove_child(link)) {
            link->from_socket->links.erase(link);
            link->to_socket->links.erase(link);
            auto to_node = link->to_socket->get_parent();
            auto from_node = link->from_socket->get_parent();
            bk_graph->remove_node_input(to_node->bk_node,
                    link->to_socket->get_index());
            links.erase(link);
            return true;
        } else {
            return false;
        }
    }

    bool remove_node(UiDopNode *node) {
        bk_graph->remove_node(node->bk_node);
        for (auto *socket: node->inputs) {
            for (auto *link: std::set(socket->links)) {
                remove_link(link);
            }
        }
        for (auto *socket: node->outputs) {
            for (auto *link: std::set(socket->links)) {
                remove_link(link);
            }
        }
        if (remove_child(node)) {
            nodes.erase(node);
            return true;
        } else {
            return false;
        }
    }

    UiDopNode *add_node(std::string kind) {
        auto p = add_child<UiDopNode>();
        p->bk_node = bk_graph->add_node(kind);
        p->name = p->bk_node->name;
        p->kind = p->bk_node->kind;
        nodes.insert(p);
        return p;
    }

    UiDopLink *add_link(UiDopOutputSocket *from_socket, UiDopInputSocket *to_socket) {
        auto p = add_child<UiDopLink>(from_socket, to_socket);
        auto to_node = to_socket->get_parent();
        auto from_node = from_socket->get_parent();
        bk_graph->set_node_input(to_node->bk_node, to_socket->get_index(),
                from_node->bk_node, from_socket->get_index());
        links.insert(p);
        return p;
    }

    // add a new pending link with one side linked to |socket| if no pending link
    // create a real link from current pending link's socket to the |socket| otherwise
    void add_pending_link(UiDopSocket *socket) {
        if (pending_link) {
            if (socket && pending_link->socket) {
                auto socket1 = pending_link->socket;
                auto socket2 = socket;
                auto output1 = dynamic_cast<UiDopOutputSocket *>(socket1);
                auto output2 = dynamic_cast<UiDopOutputSocket *>(socket2);
                auto input1 = dynamic_cast<UiDopInputSocket *>(socket1);
                auto input2 = dynamic_cast<UiDopInputSocket *>(socket2);
                if (output1 && input2) {
                    add_link(output1, input2);
                } else if (input1 && output2) {
                    add_link(output2, input1);
                }
            } else if (auto another = dynamic_cast<UiDopInputSocket *>(pending_link->socket); another) {
                another->clear_links();
            }
            remove_child(pending_link);
            pending_link = nullptr;

        } else if (socket) {
            pending_link = add_child<UiDopPendingLink>(socket);
        }
    }

    UiDopGraph() {
        auto c = add_node("readvdb", {100, 256});
        auto d = add_node("vdbsmooth", {450, 256});

        add_link(c->outputs[0], d->inputs[0]);

        auto btn = add_child<Button>();
        btn->text = "Apply";
        btn->on_clicked.connect([this] () {
            bk_graph->nodes.at("vdbsmooth1")->apply_func();
        });
    }

    void paint() const override {
        glColor3f(0.2f, 0.2f, 0.2f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }

    void on_event(Event_Mouse e) override {
        GraphicsView::on_event(e);

        if (e.down != true)
            return;
        if (e.btn != 0)
            return;

        auto item = item_at({cur.x, cur.y});

        if (auto node = dynamic_cast<UiDopNode *>(item); node) {
            if (pending_link) {
                auto another = pending_link->socket;
                if (dynamic_cast<UiDopInputSocket *>(another) && node->outputs.size()) {
                    add_pending_link(node->outputs[0]);
                } else if (dynamic_cast<UiDopOutputSocket *>(another) && node->inputs.size()) {
                    add_pending_link(node->inputs[0]);
                } else {
                    add_pending_link(nullptr);
                }
            }

        } else if (auto link = dynamic_cast<UiDopLink *>(item); link) {
            if (pending_link) {
                auto another = pending_link->socket;
                if (dynamic_cast<UiDopInputSocket *>(another)) {
                    add_pending_link(link->from_socket);
                } else if (dynamic_cast<UiDopOutputSocket *>(another)) {
                    add_pending_link(link->to_socket);
                } else {
                    add_pending_link(nullptr);
                }
            }

        } else if (auto socket = dynamic_cast<UiDopSocket *>(item); socket) {
            add_pending_link(socket);

        } else {
            add_pending_link(nullptr);
        }
    }

    UiDopContextMenu *menu = nullptr;

    UiDopNode *add_node(std::string kind, Point pos) {
        auto node = add_node(kind);
        node->position = pos;
        node->kind = kind;
        node->add_input_socket()->name = "path";
        node->add_input_socket()->name = "type";
        node->add_output_socket()->name = "grid";
        node->update_sockets();
        return node;
    }

    UiDopContextMenu *add_context_menu() {
        remove_context_menu();
        menu = add_child<UiDopContextMenu>();

        menu->add_entry("vdbsmooth");
        menu->add_entry("readvdb");
        menu->add_entry("vdberode");
        menu->update_entries();

        menu->on_selected.connect([this] {
            add_node(menu->selection, menu->position);
            remove_context_menu();
        });

        return menu;
    }

    void remove_context_menu() {
        if (menu) {
            remove_child(menu);
            menu = nullptr;
        }
    }

    void on_event(Event_Key e) override {
        Widget::on_event(e);

        if (e.down != true)
            return;

        if (e.key == GLFW_KEY_TAB) {
            add_context_menu();

        } else if (e.key == GLFW_KEY_DELETE) {
            for (auto *item: children_selected) {
                if (auto link = dynamic_cast<UiDopLink *>(item); link) {
                    remove_link(link);
                } else if (auto node = dynamic_cast<UiDopNode *>(item); node) {
                    remove_node(node);
                }
            }
            children_selected.clear();
            select_child(nullptr, false);
        }
    }

    UiDopEditor *editor = nullptr;

    void select_child(GraphicsWidget *ptr, bool multiselect) override;
};

bool UiDopSocket::is_parent_active() const {
    return get_parent()->hovered;
}

void UiDopSocket::clear_links() {
    auto graph = get_parent()->get_parent();
    if (links.size()) {
        for (auto link: std::set(links)) {
            graph->remove_link(link);
        }
    }
}


struct UiDopParam : Widget {
    Label *label;
    TextEdit *edit;

    UiDopParam() {
        bbox = {0, 0, 400, 50};
        label = add_child<Label>();
        label->position = {0, 5};
        label->bbox = {0, 0, 100, 40};
        edit = add_child<TextEdit>();
        edit->position = {100, 5};
        edit->bbox = {0, 0, 400, 40};
    }

    void set_bk_socket
        ( UiDopInputSocket *socket
        , DopInputSocket *bk_socket
        , DopNode *bk_node
        ) {
        label->text = socket->name;
        edit->text = bk_socket->value;
        edit->disabled = socket->links.size();

        edit->on_editing_finished.connect([=, this] {
            if (bk_socket->value != edit->text) {
                bk_socket->value = edit->text;
                bk_node->invalidate();
            }
        });
    }
};


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

void UiDopGraph::select_child(GraphicsWidget *ptr, bool multiselect) {
    GraphicsView::select_child(ptr, multiselect);
    if (editor)
        editor->set_selection(dynamic_cast<UiDopNode *>(ptr));
}

// END node editor ui

// BEG main window ui

struct RootWindow : Widget {
    UiDopGraph *graph;
    UiDopEditor *editor;

    RootWindow() {
        graph = add_child<UiDopGraph>();
        graph->bbox = {0, 0, 1024, 512};
        graph->position = {0, 256};
        editor = add_child<UiDopEditor>();
        editor->bbox = {0, 0, 1024, 256};
        graph->editor = editor;
    }
} win;

// END main window ui

// BEG ui library main loop

void process_input() {
    GLint nx = 100, ny = 100;
    glfwGetFramebufferSize(window, &nx, &ny);
    glViewport(0, 0, nx, ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, 2.f, -.001f);
    glTranslatef(-.5f, -.5f, 1.f);
    glScalef(1.f / nx, 1.f / ny, 1.f);

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    cur.on_update();
    win.bbox = {0, 0, (float)nx, (float)ny};
    win.do_update();
    win.do_update_event();
    win.after_update();
    cur.after_update();
}


void draw_graphics() {
    if (cur.need_repaint) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        win.do_paint();
        glfwSwapBuffers(window);
        cur.need_repaint = false;
    }
}


static void window_refresh_callback(GLFWwindow *window) {
    cur.need_repaint = true;
}

int main() {
    if (!glfwInit()) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    window = glfwCreateWindow(1024, 768, "Zeno Editor", nullptr, nullptr);
    glfwSetWindowPos(window, 0, 0);
    if (!window) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCharCallback(window, char_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetWindowRefreshCallback(window, window_refresh_callback);

    double lasttime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwWaitEvents();
        process_input();
        draw_graphics();
#if 0
        lasttime += 1.0 / fps;
        while (glfwGetTime() < lasttime) {
            double sleepfor = (lasttime - glfwGetTime()) * 0.75;
            int us(sleepfor / 1000000);
            std::this_thread::sleep_for(std::chrono::microseconds(us));
        }
#endif
    }

    return 0;
}

// END ui library main loop
