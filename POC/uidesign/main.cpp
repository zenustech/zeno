#include <cmath>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <variant>
#include <optional>
#include <functional>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <FTGL/ftgl.h>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <list>
#include <set>
#include <map>
#include <any>
#include "ztd/Map.h"


#define typenameof(x) typeid(*(x)).name()


// BEG generic ui library

struct Color {
    float r, g, b;

    Color(float r = 0, float g = 0, float b = 0)
        : r(r), g(g), b(b) {}

    float *data() {
        return &r;
    }

    float const *data() const {
        return &r;
    }
};

struct Point {
    float x, y;

    Point(float x = 0, float y = 0)
        : x(x), y(y) {}

    Point operator+(Point const &o) const {
        return {x + o.x, y + o.y};
    }

    Point operator-(Point const &o) const {
        return {x - o.x, y - o.y};
    }

    Point operator*(float o) const {
        return {x * o, y * o};
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



struct Event_Key {
    int key;
    int mode;
    bool down;
};

struct Event_Char {
    unsigned int code;
};

struct Event_Hover {
    bool enter;
};

struct Event_Mouse {
    int btn;  // lmb=0, mmb=1, rmb=2
    bool down;
};

struct Event_Motion {
    float x, y;
};

using Event = std::variant
    < Event_Key
    , Event_Char
    , Event_Hover
    , Event_Mouse
    , Event_Motion
    >;


struct Widget;

struct CursorState {
    float x = 0, y = 0;
    float dx = 0, dy = 0;
    float last_x = 0, last_y = 0;
    bool lmb = false, mmb = false, rmb = false;
    bool shift = false, ctrl = false, alt = false;

    void on_update() {
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
        } raii {[=, this] () {
            x = ox; y = oy;
        }};
        return raii;
    }

    std::vector<Event> events;

    bool need_repaint = true;

    void after_update() {
        if (events.size() || dx || dy)
            need_repaint = true;
        events.clear();
    }
} cur;

static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    //GLint nx, ny;
    //glfwGetFramebufferSize(window, &nx, &ny);
    //auto x = 0.5f + (float)xpos;
    //auto y = ny - 0.5f - (float)ypos;
    //cur.events.push_back(Event_Motion{.x = x, .y = y});
}

static void mouse_button_callback(GLFWwindow *window, int btn, int action, int mode) {
    cur.events.push_back(Event_Mouse{.btn = btn, .down = action == GLFW_PRESS});
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    cur.events.push_back(Event_Key{.key = key, .mode = mode, .down = action == GLFW_PRESS});
}

static void char_callback(GLFWwindow *window, unsigned int codeprint) {
    cur.events.push_back(Event_Char{.code = codeprint});
}


template <class T>
inline T notnull(T &&t) {
    if (!t) throw std::bad_optional_access();
    return t;
}

struct Object {
    Object() = default;
    Object(Object const &) = delete;
    Object &operator=(Object const &) = delete;
    Object(Object &&) = delete;
    Object &operator=(Object &&) = delete;
    virtual ~Object() = default;
};


struct Widget : Object {
    Widget *parent = nullptr;
    std::list<std::unique_ptr<Widget>> children;
    std::vector<std::unique_ptr<Widget>> children_gc;
    Point position{0, 0};
    float zvalue{0};

    template <class T, class ...Ts>
    T *add_child(Ts &&...ts) {
        auto p = std::make_unique<T>(std::forward<Ts>(ts)...);
        T *raw_p = p.get();
        p->parent = this;
        children.push_back(std::move(p));
        return raw_p;
    }

    void remove_all_children() {
        for (auto &child: children) {
            child->parent = nullptr;
            children_gc.push_back(std::move(child));
        }
    }

    bool remove_child(Widget *ptr) {
        for (auto it = children.begin(); it != children.end(); it++) {
            auto &child = *it;
            if (child.get() == ptr) {
                child->parent = nullptr;
                // transfer ownership to gc (also set child to null):
                children_gc.push_back(std::move(child));
                children.erase(it);
                return true;
            }
        }
        return false;
    }

    bool hovered = false;
    bool pressed[3] = {false, false, false};

    virtual Widget *child_at(Point p) const {
        Widget *found = nullptr;
        for (auto const &child: children) {
            if (child->contains_point(p - child->position)) {
                if (!found || child->zvalue >= found->zvalue) {
                    found = child.get();
                }
            }
        }
        return found;
    }

    virtual Widget *item_at(Point p) const {
        if (!contains_point(p)) {
            return nullptr;
        }
        Widget *found = nullptr;
        float found_zvalue = 0.0f;
        for (auto const &child: children) {
            if (auto it = child->item_at(p - child->position)) {
                auto it_zvalue = child->absolute_zvalue();
                if (!found || it_zvalue >= found_zvalue) {
                    found = it;
                    found_zvalue = it_zvalue;
                }
            }
        }
        if (found) return found;
        return const_cast<Widget *>(this);
    }

    float absolute_zvalue() const {
        return parent ? zvalue : parent->absolute_zvalue() + zvalue;
    }

    virtual void on_event(Event_Hover e) {
    }

    virtual void on_event(Event_Motion e) {
    }

    virtual void on_event(Event_Mouse e) {
        pressed[e.btn] = e.down;
    }

    virtual void on_event(Event_Key e) {
    }

    virtual void on_event(Event_Char e) {
    }

    virtual void on_generic_event(Event e) {
        std::visit([this] (auto e) {
            on_event(e);
        }, e);
    }

    AABB bbox{0, 0, 10, 10};

    virtual bool contains_point(Point p) const {
        return bbox.contains(p.x, p.y);
    }

    virtual void after_update() {
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

        for (auto const &child: children) {
            child->after_update();
        }
        children_gc.clear();
    }

    virtual void do_update_event() {
        auto raii = cur.translate(-position.x, -position.y);

        if (auto child = child_at({cur.x, cur.y}); child) {
            child->do_update_event();
        }

        for (auto const &e: cur.events) {
            on_generic_event(e);
        }

        if (cur.dx || cur.dy) {
            on_event(Event_Motion{.x = -1, .y = -1});
        }
    }

    virtual void do_update() {
        auto raii = cur.translate(-position.x, -position.y);

        auto old_hovered = hovered;
        hovered = contains_point({cur.x, cur.y});

        for (auto const &child: children) {
            child->do_update();
        }

        if (!old_hovered && hovered) {
            on_event(Event_Hover{.enter = true});
        } else if (old_hovered && !hovered) {
            on_event(Event_Hover{.enter = false});
        }
    }

    virtual void do_paint() {
        auto raii = cur.translate(-position.x, -position.y);
        glPushMatrix();
        glTranslatef(position.x, position.y, zvalue);
        paint();
        for (auto const &child: children) {
            child->do_paint();
        }
        glPopMatrix();
    }

    virtual void paint() const {}
};


struct SignalSlot {
    using Callback = std::function<void()>;
    std::vector<Callback> callbacks;

    void operator()() const {
        for (auto const &func: callbacks) {
            func();
        }
    }

    void connect(Callback &&f) {
        callbacks.push_back(std::move(f));
    }
};


struct GraphicsWidget : Widget {
    bool selected = false;
    bool selectable = false;
    bool draggable = false;
};


struct GraphicsView : Widget {
    std::set<GraphicsWidget *> children_selected;

    virtual void select_child(GraphicsWidget *ptr, bool multiselect) {
        if (!(multiselect || (ptr && ptr->selected))) {
            for (auto const &child: children_selected) {
                child->selected = false;
            }
            children_selected.clear();
        }
        if (ptr) {
            if (ptr->selected && multiselect) {
                children_selected.erase(ptr);
                ptr->selected = false;
            } else {
                children_selected.insert(ptr);
                ptr->selected = true;
            }
        }
    }

    void on_event(Event_Motion e) override {
        Widget::on_event(e);
        if (cur.lmb) {
            for (auto const &child: children_selected) {
                if (child->draggable) {
                    child->position.x += cur.dx;
                    child->position.y += cur.dy;
                }
            }
        }
    }

    void on_event(Event_Mouse e) override {
        Widget::on_event(e);

        if (e.down != true)
            return;
        if (e.btn != 0)
            return;

        if (auto item = dynamic_cast<GraphicsWidget *>(item_at({cur.x, cur.y})); item) {
            if (item->selectable)
                select_child(item, cur.shift);
        } else if (!cur.shift) {
            select_child(nullptr, false);
        }
    }
};


struct GraphicsRectItem : GraphicsWidget {
    void paint() const override {
        if (selected || pressed[0]) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};


struct Label : Widget {
    static constexpr float BW = 8.f;

    Label() {
        bbox = {0, 0, 300, 40};
    }

    std::string text;

    float font_size = 20.f;
    FTGL::TextAlignment alignment = FTGL::ALIGN_LEFT;

    void paint() const override {
        glColor3f(0.25f, 0.25f, 0.25f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        Font font("assets/regular.ttf");
        font.set_font_size(font_size);
        font.set_fixed_width(bbox.nx - BW * 2, alignment);
        font.set_fixed_height(bbox.ny);
        glColor3f(1.f, 1.f, 1.f);

        font.render(bbox.x0 + BW, bbox.y0, text);
    }
};


struct Button : Label {
    SignalSlot on_clicked;

    void on_event(Event_Mouse e) override {
        Widget::on_event(e);

        if (e.down != false)
            return;
        if (e.btn != 0)
            return;

        on_clicked();
    }

    void paint() const override {
        if (pressed[0]) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        if (text.size()) {
            Font font("assets/regular.ttf");
            font.set_font_size(font_size);
            font.set_fixed_width(bbox.nx - BW * 2, alignment);
            font.set_fixed_height(bbox.ny);
            glColor3f(1.f, 1.f, 1.f);
            font.render(bbox.x0 + BW, bbox.y0, text);
        }
    }
};


struct TextEdit : Label {
    int cursor = 0;
    int sellen = 0;

    float font_size = 20.f;

    void _insert_text(auto content) {
        text = text.substr(0, cursor) + content + text.substr(cursor + sellen);
        sellen = 0;
    }

    SignalSlot on_editing_finished;

    void on_event(Event_Hover e) override {
        Widget::on_event(e);

        on_editing_finished();
    }

    void on_event(Event_Mouse e) override {
        Widget::on_event(e);

        if (e.down != true)
            return;
        if (e.btn != 0)
            return;

        if (cursor == 0 && sellen == text.size()) {
            cursor = text.size();
            sellen = 0;
        } else {
            cursor = 0;
            sellen = text.size();
        }
    }

    void on_event(Event_Key e) override {
        Widget::on_event(e);

        if (e.down != true)
            return;

        if (e.key == GLFW_KEY_V && e.mode == GLFW_MOD_CONTROL) {
            auto str = glfwGetClipboardString(window);
            _insert_text(str);

        } else if (e.key == GLFW_KEY_C && e.mode == GLFW_MOD_CONTROL) {
            auto str = text.substr(cursor, sellen);
            if (str.size())
                glfwSetClipboardString(window, str.c_str());

        } else if (e.key == GLFW_KEY_A && e.mode == GLFW_MOD_CONTROL) {
            cursor = 0;
            sellen = text.size();

        } else if (e.key == GLFW_KEY_LEFT) {
            cursor = std::max(0, cursor - 1);
            sellen = 0;

        } else if (e.key == GLFW_KEY_RIGHT) {
            cursor = std::min((int)text.size(), cursor + 1 + sellen);
            sellen = 0;

        } else if (e.key == GLFW_KEY_BACKSPACE) {
            if (sellen) {
                _insert_text("");
            } else if (cursor - 1 > 0) {
                text = text.substr(0, cursor - 1) + text.substr(cursor);
                cursor = std::max(0, cursor - 1);
            } else {
                text = text.substr(cursor);
                cursor = std::max(0, cursor - 1);
            }

        }
    }

    void on_event(Event_Char e) override {
        Widget::on_event(e);

        char c = e.code;
        _insert_text(c);
        cursor++;
    }

    void paint() const override {
        if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        Font font("assets/regular.ttf");
        font.set_font_size(font_size);
        font.set_fixed_width(bbox.nx - BW * 2, alignment);
        font.set_fixed_height(bbox.ny);
        glColor3f(1.f, 1.f, 1.f);

        auto txt = !hovered ? text : sellen == 0
            ? text.substr(0, cursor) + '|' + text.substr(cursor)
            : text.substr(0, cursor) + '|' + text.substr(cursor, sellen) + '|' + text.substr(cursor + sellen);
        font.render(bbox.x0 + BW, bbox.y0, txt);
    }
};

// END generic ui library

// BEG node data structures

struct DopInputSocket {
    std::string name;
    std::string value;

    void serialize(std::ostream &ss) const {
        ss << name << "=" << value;
    }
};

struct DopOutputSocket {
    std::string name;
    std::any result;

    void serialize(std::ostream &ss) const {
        ss << name;
    }
};

using DopFunctor = std::function<std::vector<std::any>(std::vector<std::any>)>;

struct DopTable {
    struct Impl {
        ztd::Map<std::string, DopFunctor> funcs;
    };
    mutable std::unique_ptr<Impl> impl;

    Impl *get_impl() const {
        if (!impl) impl = std::make_unique<Impl>();
        return impl.get();
    }

    auto const &lookup(std::string const &kind) const {
        return get_impl()->funcs.at(kind);
    }

    int define(std::string const &kind, DopFunctor &&func) {
        get_impl()->funcs.emplace(kind, std::move(func));
        return 1;
    }
} tab;

static int def_readvdb = tab.define("readvdb",
[] (std::vector<std::any> const &args) -> std::vector<std::any> {
    printf("vdbsmooth\n");
    return {1};
});

static int def_vdbsmooth = tab.define("vdbsmooth",
[] (std::vector<std::any> const &args) -> std::vector<std::any> {
    printf("vdbsmooth\n");
    return {2};
});

static int def_vdberode = tab.define("vdberode",
[] (std::vector<std::any> const &args) -> std::vector<std::any> {
    printf("vdbsmooth\n");
    return {3};
});


struct DopGraph;

struct DopNode {
    DopGraph *graph = nullptr;

    std::string name;
    std::string kind;
    std::vector<DopInputSocket> inputs;
    std::vector<DopOutputSocket> outputs;
    bool applied = false;

    void apply_func();

    std::any get_output_by_name(std::string name) {
        if (!applied) {
            apply_func();
        }
        for (int i = 0; i < outputs.size(); i++) {
            if (outputs[i].name == name)
                return outputs[i].result;
        }
        throw ztd::Exception(ztd::toString("Bad output socket name: ", name));
    }

    void serialize(std::ostream &ss) const {
        ss << "DopNode[" << '\n';
        ss << "  name=" << name << '\n';
        ss << "  kind=" << kind << '\n';
        ss << "  inputs=[" << '\n';
        for (auto const &input: inputs) {
            ss << "    ";
            input.serialize(ss);
            ss << '\n';
        }
        ss << "  ]" << '\n';
        ss << "  outputs=[" << '\n';
        for (auto const &output: outputs) {
            ss << "    ";
            output.serialize(ss);
            ss << '\n';
        }
        ss << "  ]" << '\n';
        ss << "]" << '\n';
    }
};


struct DopGraph {
    ztd::Map<std::string, std::unique_ptr<DopNode>> nodes;

    DopNode *add_node(std::string kind) {
        auto p = std::make_unique<DopNode>();
        p->graph = this;
        auto name = _determine_name(kind);
        p->kind = kind;
        p->name = name;
        auto raw = p.get();
        nodes.emplace(name, std::move(p));
        return raw;
    }

    std::string _determine_name(std::string kind) {
        for (int i = 1; i <= 256; i++) {
            auto name = kind + std::to_string(i);
            if (!nodes.contains(name)) {
                return name;
            }
        }
        return kind + std::to_string(abs(rand() * RAND_MAX + rand())) + 'a';
    }

    bool remove_node(DopNode *node) {
        for (auto const &[k, n]: nodes) {
            if (n.get() == node) {
                nodes.erase(k);
                return true;
            }
        }
        return false;
    }

    void serialize(std::ostream &ss) const {
        for (auto const &[k, node]: nodes) {
            node->serialize(ss);
            ss << '\n';
        }
    }

    static void set_node_input
        ( DopNode *to_node
        , int to_socket_index
        , DopNode *from_node
        , int from_socket_index
        )
    {
        auto const &from_socket = from_node->outputs.at(from_socket_index);
        auto &to_socket = to_node->inputs.at(to_socket_index);
        auto refid = '@' + from_node->name + ':' + from_socket.name;
        to_socket.value = refid;
    }

    static void remove_node_input
        ( DopNode *to_node
        , int to_socket_index
        )
    {
        auto &to_socket = to_node->inputs.at(to_socket_index);
        to_socket.value = {};
    }

    std::any resolve_value(std::string expr) {
        if (expr[0] == '@') {
            auto i = expr.find(':');
            auto node_n = expr.substr(1, i - 1);
            auto socket_n = expr.substr(i + 1);
            auto *node = nodes.at(node_n).get();
            return node->get_output_by_name(socket_n);
        } else if (expr.size()) {
            return std::stoi(expr);
        } else {
            return {};
        }
    }
};

void DopNode::apply_func() {
    auto func = tab.lookup(kind);
    std::vector<std::any> input_vals;
    for (auto const &input: inputs) {
        input_vals.push_back(graph->resolve_value(input.value));
    }
    auto output_vals = func(input_vals);
    for (int i = 0; i < std::min(outputs.size(), output_vals.size()); i++) {
        outputs[i].result = std::move(output_vals[i]);
    }
    applied = true;
}

// END node data structures

// BEG node editor ui

struct UiDopLink;
struct UiDopNode;

struct UiDopSocket : GraphicsRectItem {
    static constexpr float BW = 4, R = 15, FH = 19, NW = 200;

    std::string name;
    std::set<UiDopLink *> links;

    UiDopSocket() {
        bbox = {-R, -R, 2 * R, 2 * R};
        zvalue = 2.f;
    }

    void paint() const override {
        glColor3f(0.75f, 0.75f, 0.75f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
        if (hovered) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (links.size()) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0 + BW, bbox.y0 + BW, bbox.x0 + bbox.nx - BW, bbox.y0 + bbox.ny - BW);
    }

    UiDopNode *get_parent() const {
        return (UiDopNode *)(parent);
    }

    bool is_parent_active() const;
    void clear_links();
};


struct UiDopInputSocket : UiDopSocket {
    void paint() const override {
        UiDopSocket::paint();

        if (is_parent_active()) {
            Font font("assets/regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_LEFT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(R * 1.3f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(UiDopLink *link) {
        clear_links();
        links.insert(link);
    }

    int get_index() const;
};


struct UiDopOutputSocket : UiDopSocket {
    void paint() const override {
        UiDopSocket::paint();

        if (is_parent_active()) {
            Font font("assets/regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_RIGHT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(-NW - R * 1.5f, -R + FH * 0.15f, name);
        }
    }

    void attach_link(UiDopLink *link) {
        links.insert(link);
    }

    int get_index() const;
};


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

int UiDopInputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->inputs.size(); i++) {
        if (node->inputs[i] == this) {
            return i;
        }
    }
    throw ztd::Exception("Cannot find index of input node");
}

int UiDopOutputSocket::get_index() const {
    auto node = get_parent();
    for (int i = 0; i < node->outputs.size(); i++) {
        if (node->outputs[i] == this) {
            return i;
        }
    }
    throw ztd::Exception("Cannot find index of output node");
}


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

    void set_bk_socket(UiDopInputSocket *socket, DopInputSocket *bk_socket) {
        label->text = socket->name;
        edit->text = bk_socket->value;

        edit->on_editing_finished.connect([=, this] {
            if (!socket->links.size()) {
                bk_socket->value = edit->text;
            } else {
                edit->text = bk_socket->value;
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
                param->set_bk_socket(socket, bk_socket);
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
