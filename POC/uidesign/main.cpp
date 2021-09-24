#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unistd.h>
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
        font = std::make_unique<FTPolygonFont>(path);
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
    bool shift = false, ctrl = false, alt = false;

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
    }

    auto translate(float dx, float dy) {
        x += dx; y += dy;
        struct RAII : std::function<void()> {
            using std::function<void()>::function;
            ~RAII() { (*this)(); }
        } raii {[=] () {
            x -= dx; y -= dy;
        }};
        return raii;
    }
} cur;


struct IWidget {
    virtual ~IWidget() = default;

    virtual void do_update() = 0;
    virtual void do_paint() = 0;
};


struct Widget : IWidget {
    Widget *parent = nullptr;
    std::vector<std::unique_ptr<Widget>> children;
    Point position{0, 0};

    Widget() = default;
    Widget(Widget const &) = delete;
    Widget &operator=(Widget const &) = delete;

    template <class T, class ...Ts>
    T *add_child(Ts &&...ts) {
        std::unique_ptr<Widget> p = std::make_unique<T>(std::forward<Ts>(ts)...);
        p->parent = this;
        auto raw_p = p.get();
        children.push_back(std::move(p));
        return static_cast<T *>(raw_p);
    }

    virtual AABB get_bounding_box() const = 0;

    virtual void on_hover_enter() {
    }

    virtual void on_hover_leave() {
        if (lmb_pressed) on_lmb_up();
        if (mmb_pressed) on_mmb_up();
        if (rmb_pressed) on_rmb_up();
    }

    std::set<Widget *> children_selected;

    bool hovered = false;

    virtual void on_mouse_move() {
    }

    virtual void on_lmb_down() {
        lmb_pressed = true;
    }

    virtual void on_lmb_up() {
        lmb_pressed = false;
    }

    virtual void on_mmb_down() {
        mmb_pressed = true;
    }

    virtual void on_mmb_up() {
        mmb_pressed = false;
    }

    virtual void on_rmb_down() {
        rmb_pressed = true;
    }

    virtual void on_rmb_up() {
        rmb_pressed = false;
    }

    bool lmb_pressed = false;
    bool mmb_pressed = false;
    bool rmb_pressed = false;

    void do_update() override {
        auto raii = cur.translate(-position.x, -position.y);
        auto bbox = get_bounding_box();

        auto old_hovered = hovered;
        hovered = bbox.contains(cur.x, cur.y);

        for (auto const &child: children) {
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

    void do_paint() override {
        glPushMatrix();
        glTranslatef(position.x, position.y, 0.f);
        paint();
        for (auto const &child: children) {
            child->do_paint();
        }
        glPopMatrix();
    }

    virtual void paint() const {}
};


struct GraphicsWidget : Widget {
    std::set<GraphicsWidget *> children_selected;

    bool selected = false;
    bool selectable = false;
    bool draggable = false;

    void _select_child(GraphicsWidget *ptr, bool multiselect = false) {
        if (!(multiselect || (ptr && ptr->selected))) {
            for (auto *child: children_selected) {
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

    void on_mouse_move() override {
        if (cur.lmb) {
            for (auto *child: children_selected) {
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


struct DopSocket : GraphicsRectItem {
    static constexpr float BW = 4, R = 15, FH = 18, NW = 200;

    std::string title = "(untitled)";
    bool connected = false;

    void paint() const override {
        glColor3f(0.75f, 0.75f, 0.75f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
        if (connected) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0 + BW, bbox.y0 + BW, bbox.x0 + bbox.nx - BW, bbox.y0 + bbox.ny - BW);
    }
};


struct DopInputSocket : DopSocket {
    DopInputSocket() {
        set_bounding_box({0, -R, 2 * R, 2 * R});
    }

    void paint() const override {
        DopSocket::paint();

        if (hovered || lmb_pressed) {
            Font font("LiberationMono-Regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_LEFT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(R * 2.3f, -R + FH * 0.15f, title);
        }
    }
};


struct DopOutputSocket : DopSocket {
    DopOutputSocket() {
        set_bounding_box({-2 * R, -R, 2 * R, 2 * R});
    }

    void paint() const override {
        DopSocket::paint();

        if (hovered || lmb_pressed) {
            Font font("LiberationMono-Regular.ttf");
            font.set_font_size(FH);
            font.set_fixed_height(2 * R);
            font.set_fixed_width(NW, FTGL::ALIGN_RIGHT);
            glColor3f(1.f, 1.f, 1.f);
            font.render(-NW - R * 2.5f, -R + FH * 0.15f, title);
        }
    }
};


struct DopNode : GraphicsRectItem {
    static constexpr float DH = 42, TH = 42, FH = 25, W = 200, BW = 3;

    std::vector<DopInputSocket *> inputs;
    std::vector<DopOutputSocket *> outputs;
    std::string title = "(untitled)";

    void _update_input_positions() {
        for (int i = 0; i < inputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            inputs[i]->position = {0, -y};
        }
        _update_node_height();
    }

    void _update_output_positions() {
        for (int i = 0; i < outputs.size(); i++) {
            auto y = DH * (i + 0.5f);
            outputs[i]->position = {W, -y};
        }
        _update_node_height();
    }

    void _update_node_height() {
        auto h = std::max(outputs.size(), inputs.size()) * DH;
        set_bounding_box({0, -h, W, h + TH});
    }

    DopInputSocket *add_input_socket() {
        auto p = add_child<DopInputSocket>();
        inputs.push_back(p);
        _update_input_positions();
        return p;
    }

    DopOutputSocket *add_output_socket() {
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
        font.render(0, FH * 0.05f, title);
    }
};


struct NodeEditor : GraphicsRectItem {
    NodeEditor() {
        set_bounding_box({0, 0, 550, 400});

        auto c = add_child<DopNode>();
        c->position = {50, 300};
        c->add_input_socket()->title = "path";
        c->add_input_socket()->title = "type";
        c->add_output_socket()->title = "grid";
        c->title = "readvdb";

        auto d = add_child<DopNode>();
        d->position = {300, 300};
        d->add_input_socket()->title = "grid";
        d->add_input_socket()->title = "width";
        d->add_input_socket()->title = "times";
        d->add_output_socket()->title = "grid";
        d->title = "vdbsmooth";
    }

    void paint() const override {
        glColor3f(0.2f, 0.2f, 0.2f);
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};


struct RootWindow : Widget {
    RootWindow() {
        add_child<NodeEditor>();
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
    glClearColor(0.2f, 0.3f, 0.5f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    win.do_paint();
}


int main() {
    if (!glfwInit()) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    window = glfwCreateWindow(800, 600, "Zeno Editor", nullptr, nullptr);
    if (!window) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to create GLFW window: %s\n", err);
        return -1;
    }
    glfwMakeContextCurrent(window);

    while (!glfwWindowShouldClose(window)) {
        process_input();
        draw_graphics();
        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(16000);
    }

    return 0;
}
