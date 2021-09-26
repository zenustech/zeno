#include <cmath>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <functional>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <FTGL/ftgl.h>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <set>

struct Point {
    float x, y;

    Point(float x = 0, float y = 0)
        : x(x), y(y) {}

    Point operator+(Point const &o) const {
        return {x + o.x, y + o.y};
    }
};

struct AABB {
    float x0, y0, nx, ny;

    AABB(float x0 = 0, float y0 = 0, float nx = 0, float ny = 0)
        : x0(x0), y0(y0), nx(nx), ny(ny) {}

    bool contains(float x, float y) const {
        return x0 <= x && y0 <= y && x <= x0 + nx && y <= y0 + ny;
    }
};


struct Font {
    std::unique_ptr<FTFont> font;
    std::unique_ptr<FTSimpleLayout> layout;
    float fixed_height = -1;

    Font(const char *path) {
        font = std::make_unique<FTTextureFont>(path);
        if (font->Error()) {
            fprintf(stderr, "Failed to load font: %s\n", path);
            abort();
        }
        font->CharMap(ft_encoding_unicode);

        layout = std::make_unique<FTSimpleLayout>();
        layout->SetFont(font.get());
    }

    Font &set_font_size(float font_size) {
        font->FaceSize(font_size);
        return *this;
    }

    Font &set_fixed_width(float width, FTGL::TextAlignment align = FTGL::ALIGN_CENTER) {
        layout->SetLineLength(width);
        layout->SetAlignment(align);
        return *this;
    }

    Font &set_fixed_height(float height) {
        fixed_height = height;
        return *this;
    }

    AABB calc_bounding_box(std::string const &str) {
        auto bbox = layout->BBox(str.data(), str.size());
        return AABB(bbox.Lower().X(), bbox.Lower().Y(),
                    bbox.Upper().X() - bbox.Lower().X(),
                    bbox.Upper().Y() - bbox.Lower().Y());
    }

    Font &render(float x, float y, std::string const &str) {
        if (fixed_height > 0) {
            auto bbox = calc_bounding_box(str);
            y += fixed_height / 2 - bbox.ny / 2;
        }
        if (str.size()) {
            glPushMatrix();
            glTranslatef(x, y, 0.f);
            layout->Render(str.data(), str.size());
            glPopMatrix();
        }
        return *this;
    }
};


GLFWwindow *window;


struct Widget;

struct CursorState {
    float x = 0, y = 0;
    float dx = 0, dy = 0;
    float last_x = 0, last_y = 0;
    bool lmb = false, mmb = false, rmb = false;
    bool last_lmb = false, last_mmb = false, last_rmb = false;
    bool shift = false, ctrl = false, alt = false, del = false;

    void on_update() {
        last_lmb = lmb;
        last_mmb = mmb;
        last_rmb = rmb;
        last_x = x;
        last_y = y;

        GLint nx, ny;
        glfwGetFramebufferSize(window, &nx, &ny);
        GLdouble _x, _y;
        glfwGetCursorPos(window, &_x, &_y);
        x = 0.5f + (float)_x;
        y = ny - 0.5f - (float)_y;
        dx = x - last_x;
        dy = y - last_y;
        lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
        ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
        alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS;
        del = glfwGetKey(window, GLFW_KEY_DELETE) == GLFW_PRESS;
    }

    bool is_pressed(int key) const {
        return glfwGetKey(window, key) == GLFW_PRESS;
    }

    auto translate(float dx, float dy) {
        auto ox = x, oy = y;
        x += dx; y += dy;
        struct RAII : std::function<void()> {
            using std::function<void()>::function;
            ~RAII() { (*this)(); }
        } raii {[=] () {
            x = ox; y = oy;
        }};
        return raii;
    }
} cur;


#if 0
struct Object {
    int refcnt = 1;

    Object() = default;
    Object(Object const &) = delete;
    Object &operator=(Object const &) = delete;
    Object(Object &&) = delete;
    Object &operator=(Object &&) = delete;

    virtual ~Object() = default;
};

template <class T>
struct Ptr {
    T *ptr = nullptr;

    Ptr(T *ptr) : ptr(ptr) {}

    Ptr &operator=(T *nptr) {
        release();
        ptr = nptr;
        return *this;
    }

    Ptr(Ptr const &p) : ptr(p.ptr) {
        if (ptr)
            ptr->refcnt++;
    }

    Ptr &operator=(Ptr const &p) {
        ptr = p.ptr;
        if (ptr)
            ptr->refcnt++;
        return *this;
    }

    Ptr(Ptr &&p) : ptr(p.ptr) {
        p.ptr = nullptr;
    }

    Ptr &operator=(Ptr &&p) {
        ptr = p.ptr;
        p.ptr = nullptr;
        return *this;
    }

    T *get() const {
        return ptr;
    }

    template <class S>
    Ptr<S> cast() const {
        if (ptr)
            ptr->refcnt++;
        return Ptr<S>(dynamic_cast<S *>(ptr));
    }

    template <class S>
    operator Ptr<S>() const {
        if (ptr)
            ptr->refcnt++;
        return Ptr<S>(static_cast<S *>(ptr));
    }

    operator T *() const {
        return get();
    }

    T &operator*() const {
        return *get();
    }

    T *operator->() const {
        return get();
    }

    void release() {
        if (ptr && --ptr->refcnt <= 0) {
            delete ptr;
            ptr = nullptr;
        }
    }

    Ptr(std::nullptr_t = nullptr) : ptr(nullptr) {
    }

    Ptr &operator=(std::nullptr_t) {
        release();
        return *this;
    }

    ~Ptr() {
        release();
    }
};

template <class T>
Ptr(T *) -> Ptr<T>;

template <class T, class ...Ts>
Ptr<T> makePtr(Ts &&...ts) {
    return new T(std::forward<Ts>(ts)...);
}

#else

struct Object {
    Object() = default;
    Object(Object const &) = delete;
    Object &operator=(Object const &) = delete;
    Object(Object &&) = delete;
    Object &operator=(Object &&) = delete;

    virtual ~Object() = default;
};

template <class T>
struct Ptr : std::shared_ptr<T> {
    using std::shared_ptr<T>::shared_ptr;

    Ptr(std::shared_ptr<T> &&p) : std::shared_ptr<T>(std::move(p)) {}
    Ptr(std::shared_ptr<T> const &p) : std::shared_ptr<T>(p) {}
    Ptr(T *p) : std::shared_ptr<T>(p) {}
    operator T *() const { return this->get(); }
};

template <class T>
Ptr(T *) -> Ptr<T>;

template <class T>
Ptr(std::shared_ptr<T>) -> Ptr<T>;

template <class T, class ...Ts>
Ptr<T> makePtr(Ts &&...ts) {
    return std::make_shared<T>(std::forward<Ts>(ts)...);
}

#endif


struct Widget : Object {
    Widget *parent = nullptr;
    std::vector<Ptr<Widget>> children;
    Point position{0, 0};

    template <class T, class ...Ts>
    Ptr<T> add_child(Ts &&...ts) {
        auto p = makePtr<T>(std::forward<Ts>(ts)...);
        p->parent = this;
        children.push_back(p);
        invalidate();
        return p;
    }

    bool remove_child(Widget *ptr) {
        for (auto &child: children) {
            if (child == ptr) {
                ptr->parent = nullptr;
                invalidate();
                child = nullptr;
                return true;
            }
        }
        return false;
    }

    virtual AABB get_bounding_box() const = 0;

    virtual void on_hover_enter() {
        invalidate();
    }

    virtual void on_hover_leave() {
        invalidate();
        if (lmb_pressed) on_lmb_up();
        if (mmb_pressed) on_mmb_up();
        if (rmb_pressed) on_rmb_up();
    }

    bool hovered = false;

    virtual void on_mouse_move() {
        invalidate();
    }

    virtual void on_lmb_down() {
        invalidate();
        lmb_pressed = true;
    }

    virtual void on_lmb_up() {
        invalidate();
        lmb_pressed = false;
    }

    virtual void on_mmb_down() {
        invalidate();
        mmb_pressed = true;
    }

    virtual void on_mmb_up() {
        invalidate();
        mmb_pressed = false;
    }

    virtual void on_rmb_down() {
        invalidate();
        rmb_pressed = true;
    }

    virtual void on_rmb_up() {
        invalidate();
        rmb_pressed = false;
    }

    bool lmb_pressed = false;
    bool mmb_pressed = false;
    bool rmb_pressed = false;

    void _clean_null_children() {
        bool has_any;
        do {
            has_any = false;
            for (auto it = children.begin(); it != children.end(); it++) {
                if (!*it) {
                    children.erase(it);
                    has_any = true;
                    break;
                }
            }
        } while (has_any);
    }

    virtual void do_update() {
        auto raii = cur.translate(-position.x, -position.y);
        auto bbox = get_bounding_box();

        auto old_hovered = hovered;
        hovered = bbox.contains(cur.x, cur.y);

        for (auto const &child: children) {
            if (child)
                child->do_update();
        }

        if (hovered) {
            if (!cur.last_lmb && cur.lmb) {
                on_lmb_down();
                cur.last_lmb = cur.lmb;
            }
            if (!cur.last_mmb && cur.mmb) {
                on_mmb_down();
                cur.last_mmb = cur.mmb;
            }
            if (!cur.last_rmb && cur.rmb) {
                on_rmb_down();
                cur.last_rmb = cur.rmb;
            }
        }

        if (cur.dx || cur.dy) {
            on_mouse_move();
        }

        if (hovered) {
            if (cur.last_lmb && !cur.lmb) {
                on_lmb_up();
                cur.last_lmb = cur.lmb;
            }
            if (cur.last_mmb && !cur.mmb) {
                on_mmb_up();
                cur.last_mmb = cur.mmb;
            }
            if (cur.last_rmb && !cur.rmb) {
                on_rmb_up();
                cur.last_rmb = cur.rmb;
            }
        }

        if (!old_hovered && hovered) {
            on_hover_enter();
        } else if (old_hovered && !hovered) {
            on_hover_leave();
        }
    }

    bool invalid = true;

    void invalidate() {
        if (!invalid) {
            invalid = true;
            if (parent)
                parent->invalidate();
            for (auto const &child: children) {
                if (child)
                    child->invalidate();
            }
        }
    }

    virtual void do_paint() {
        if (!invalid) return;

        auto raii = cur.translate(-position.x, -position.y);
        glPushMatrix();
        glTranslatef(position.x, position.y, 0.f);
        paint();
        for (auto const &child: children) {
            if (child)
                child->do_paint();
        }
        glPopMatrix();
        invalid = false;
    }

    virtual void paint() const {}
};


struct GraphicsWidget : Widget {
    std::set<Ptr<GraphicsWidget>> children_selected;

    bool selected = false;
    bool selectable = false;
    bool draggable = false;

    void _select_child(GraphicsWidget *ptr, bool multiselect = false) {
        if (!(multiselect || (ptr && ptr->selected))) {
            for (auto const &child: children_selected) {
                child->selected = false;
            }
            children_selected.clear();
            invalidate();
        }
        if (ptr) {
            if (ptr->selected && multiselect) {
                children_selected.erase(ptr);
                ptr->selected = false;
            } else {
                children_selected.insert(ptr);
                ptr->selected = true;
            }
            invalidate();
        }
    }

    void on_mouse_move() override {
        Widget::on_mouse_move();
        if (cur.lmb) {
            for (auto const &child: children_selected) {
                if (child->draggable) {
                    child->position.x += cur.dx;
                    child->position.y += cur.dy;
                }
            }
        }
    }

    void on_lmb_down() override {
        Widget::on_lmb_down();
        if (auto par = dynamic_cast<GraphicsWidget *>(parent); par) {
            if (selectable) {
                par->_select_child(this, cur.shift);
            }
        }
        if (!cur.shift)
            _select_child(nullptr);
    }
};


struct GraphicsRectItem : GraphicsWidget {
    AABB bbox{0, 0, 200, 150};

    void set_bounding_box(AABB bbox) {
        this->bbox = bbox;
    }

    AABB get_bounding_box() const override {
        return bbox;
    }

    void paint() const override {
        if (selected || lmb_pressed) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};


struct Button : Widget {
    AABB bbox{0, 0, 150, 50};
    std::string text;

    void set_bounding_box(AABB bbox) {
        this->bbox = bbox;
    }

    AABB get_bounding_box() const override {
        return bbox;
    }

    virtual void on_clicked() {}

    void on_lmb_down() override {
        Widget::on_lmb_down();
        on_clicked();
    }

    void paint() const override {
        if (lmb_pressed) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        if (text.size()) {
            Font font("LiberationMono-Regular.ttf");
            //Font font("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc");
            font.set_font_size(30.f);
            font.set_fixed_width(bbox.nx);
            font.set_fixed_height(bbox.ny);
            glColor3f(1.f, 1.f, 1.f);
            font.render(bbox.x0, bbox.y0, text);
        }
    }
};


struct DopLink;
struct DopNode;

struct DopSocket : GraphicsRectItem {
    static constexpr float BW = 4, R = 15, FH = 19, NW = 200;

    std::string name = "(unnamed)";
    std::vector<DopLink *> links;

    DopSocket() {
        set_bounding_box({-R, -R, 2 * R, 2 * R});
    }

    void paint() const override {
        glColor3f(0.75f, 0.75f, 0.75f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
        if (links.size()) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0 + BW, bbox.y0 + BW, bbox.x0 + bbox.nx - BW, bbox.y0 + bbox.ny - BW);
    }

    DopNode *get_parent() const {
        return (DopNode *)(parent);
    }

    void on_lmb_down() override;
};


struct DopInputSocket : DopSocket {
    void paint() const override {
        DopSocket::paint();

        if (hovered || lmb_pressed) {
            Font font("LiberationMono-Regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_LEFT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(R * 1.3f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(DopLink *link);
};


struct DopOutputSocket : DopSocket {
    void paint() const override {
        DopSocket::paint();

        if (hovered || lmb_pressed) {
            Font font("LiberationMono-Regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_RIGHT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(-NW - R * 1.5f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(DopLink *link) {
        links.push_back(link);
    }
};


struct DopGraph;

struct DopNode : GraphicsRectItem {
    static constexpr float DH = 40, TH = 42, FH = 24, W = 200, BW = 3;

    std::vector<Ptr<DopInputSocket>> inputs;
    std::vector<Ptr<DopOutputSocket>> outputs;
    std::string name = "(unnamed)";
    std::string kind = "(untyped)";

    void _update_input_positions() {
        for (int i = 0; i < inputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            inputs[i]->position = {DopSocket::R, -y};
        }
        _update_node_height();
    }

    void _update_output_positions() {
        for (int i = 0; i < outputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            outputs[i]->position = {W - DopSocket::R, -y};
        }
        _update_node_height();
    }

    void _update_node_height() {
        auto h = std::max(outputs.size(), inputs.size()) * DH;
        set_bounding_box({0, -h, W, h + TH});
    }

    Ptr<DopInputSocket> add_input_socket() {
        auto p = add_child<DopInputSocket>();
        inputs.push_back(p);
        _update_input_positions();
        return p;
    }

    Ptr<DopOutputSocket> add_output_socket() {
        auto p = add_child<DopOutputSocket>();
        outputs.push_back(p);
        _update_output_positions();
        return p;
    }

    DopNode() {
        selectable = true;
        draggable = true;

        _update_node_height();
    }

    DopGraph *get_parent() const {
        return (DopGraph *)(parent);
    }

    void paint() const override {
        if (selected || lmb_pressed) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.125f, 0.375f, 0.425f);
        }
        glRectf(bbox.x0 - BW, bbox.y0 - BW, bbox.x0 + bbox.nx + BW, bbox.y0 + bbox.ny + BW);

        glColor3f(0.375f, 0.375f, 0.375f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        if (selected || lmb_pressed) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.125f, 0.375f, 0.425f);
        }
        glRectf(0.f, 0.f, W, TH);

        Font font("LiberationMono-Regular.ttf");
        font.set_font_size(FH);
        font.set_fixed_width(W);
        font.set_fixed_height(TH);
        glColor3f(1.f, 1.f, 1.f);
        font.render(0, FH * 0.05f, name);
    }

    void on_lmb_down() override;
};


struct GraphicsLineItem : GraphicsWidget {
    static constexpr float LW = 4.f;

    virtual Point get_from_position() const = 0;
    virtual Point get_to_position() const = 0;

    AABB get_bounding_box() const override {
        auto [sx, sy] = get_from_position();
        auto [dx, dy] = get_to_position();
        return {std::min(sx, dx) - LW, std::min(sy, dy) - LW, std::fabs(sx - dx) + 2 * LW, std::fabs(sy - dy) + 2 * LW};
    }

    void paint() const override {
        if (selected) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.125f, 0.375f, 0.425f);
        }
        auto [sx, sy] = get_from_position();
        auto [dx, dy] = get_to_position();
        glLineWidth(LW);
        glBegin(GL_LINE_STRIP);
        glVertex2f(sx, sy);
        glVertex2f(dx, dy);
        glEnd();
    }
};


struct DopLink : GraphicsLineItem {
    DopOutputSocket *from_socket;
    DopInputSocket *to_socket;

    DopLink(DopOutputSocket *from_socket, DopInputSocket *to_socket)
        : from_socket(from_socket), to_socket(to_socket)
    {
        from_socket->attach_link(this);
        to_socket->attach_link(this);
        selectable = true;
    }

    Point get_from_position() const override {
        return from_socket->position + from_socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return to_socket->position + to_socket->get_parent()->position;
    }

    DopGraph *get_parent() const {
        return (DopGraph *)(parent);
    }
};


struct DopPendingLink : GraphicsLineItem {
    DopSocket *socket;

    DopPendingLink(DopSocket *socket)
        : socket(socket)
    {}

    Point get_from_position() const override {
        return socket->position + socket->get_parent()->position;
    }

    Point get_to_position() const override {
        return {cur.x, cur.y};
    }

    DopGraph *get_parent() const {
        return (DopGraph *)(parent);
    }

    void on_lmb_down() override;
};


struct DopGraph : GraphicsRectItem {
    std::set<Ptr<DopNode>> nodes;
    std::set<Ptr<DopLink>> links;
    Ptr<DopPendingLink> pending_link = nullptr;

    bool remove_link(DopLink *link) {
        if (remove_child(link)) {
            links.erase(link);
            return true;
        } else {
            return false;
        }
    }

    bool remove_node(DopNode *node) {
        if (remove_child(node)) {
            nodes.erase(node);
            return true;
        } else {
            return false;
        }
    }

    Ptr<DopNode> add_node() {
        auto p = add_child<DopNode>();
        nodes.insert(p);
        return p;
    }

    Ptr<DopLink> add_link(DopOutputSocket *from_socket, DopInputSocket *to_socket) {
        auto p = add_child<DopLink>(from_socket, to_socket);
        links.insert(p);
        return p;
    }

    void add_pending_link(DopSocket *socket) {
        if (pending_link) {
            if (socket && pending_link->socket) {
                auto socket1 = pending_link->socket;
                auto socket2 = socket;
                auto output1 = dynamic_cast<DopOutputSocket *>(socket1);
                auto output2 = dynamic_cast<DopOutputSocket *>(socket2);
                auto input1 = dynamic_cast<DopInputSocket *>(socket1);
                auto input2 = dynamic_cast<DopInputSocket *>(socket2);
                if (output1 && input2) {
                    add_link(output1, input2);
                } else if (input1 && output2) {
                    add_link(output2, input1);
                }
            }
            remove_child(pending_link);
            pending_link = nullptr;

        } else if (socket) {
            pending_link = add_child<DopPendingLink>(socket);

        } else {
            remove_child(pending_link);
            pending_link = nullptr;
        }
    }

    DopGraph() {
        set_bounding_box({0, 0, 550, 400});

        auto c = add_node();
        c->position = {50, 300};
        c->add_input_socket()->name = "path";
        c->add_input_socket()->name = "type";
        c->add_output_socket()->name = "grid";
        c->name = "readvdb1";
        c->kind = "readvdb";

        auto d = add_node();
        d->position = {300, 300};
        d->add_input_socket()->name = "grid";
        d->add_input_socket()->name = "width";
        d->add_input_socket()->name = "times";
        d->add_output_socket()->name = "grid";
        d->name = "vdbsmooth1";
        d->kind = "vdbsmooth";

        add_link(c->outputs[0], d->inputs[0]);
    }

    void paint() const override {
        glColor3f(0.2f, 0.2f, 0.2f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};


void DopPendingLink::on_lmb_down() {
    GraphicsWidget::on_lmb_down();
    auto graph = get_parent();
    graph->add_pending_link(nullptr);
}


void DopNode::on_lmb_down() {
    GraphicsWidget::on_lmb_down();
    auto graph = get_parent();
    if (graph->pending_link) {
        auto another = graph->pending_link->socket;
        if (dynamic_cast<DopInputSocket *>(another) && outputs.size()) {
            graph->add_pending_link(outputs[0]);
        } else if (dynamic_cast<DopOutputSocket *>(another) && inputs.size()) {
            graph->add_pending_link(inputs[0]);
        } else {
            graph->add_pending_link(nullptr);
        }
    }
}


void DopInputSocket::attach_link(DopLink *link) {
    auto graph = get_parent()->get_parent();
    if (links.size()) {
        for (auto link: links) {
            graph->remove_link(link);
        }
    }
    links.push_back(link);
}


void DopSocket::on_lmb_down() {
    GraphicsWidget::on_lmb_down();
    auto graph = get_parent()->get_parent();
    graph->add_pending_link(this);
}


struct RootWindow : Widget {
    RootWindow() {
        add_child<DopGraph>();
    }

    AABB get_bounding_box() const override {
        return {0, 0, 800, 600};
    }
} win;


void process_input() {
    GLint nx = 100, ny = 100;
    glfwGetFramebufferSize(window, &nx, &ny);
    glViewport(0, 0, nx, ny);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(2.f, 2.f, 1.f);
    glTranslatef(-.5f, -.5f, 0.f);
    glScalef(1.f / nx, 1.f / ny, 1.f);

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    cur.on_update();
    win.position = {100, 100};
    win.do_update();
}


void draw_graphics() {
    if (win.invalid) {
        win.do_paint();
        glFlush();
    }
}


int main() {
    if (!glfwInit()) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
    window = glfwCreateWindow(800, 600, "Zeno Editor", nullptr, nullptr);
    if (!window) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(window);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POLYGON_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    double fps = 144;
    double lasttime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        process_input();
        draw_graphics();
        glfwPollEvents();
        if (fps > 0) {
            lasttime += 1.0 / fps;
            while (glfwGetTime() < lasttime) {
                double sleepfor = (lasttime - glfwGetTime()) * 0.75;
                int us(sleepfor / 1000000);
#if defined(__linux__)
                ::usleep(us);
#else
                std::this_thread::sleep_for(std::chrono::microseconds(us));
#endif
            }
        }
    }

    return 0;
}
