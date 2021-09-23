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
#include <tuple>

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
    bool lmb = false, mmb = false, rmb = false;
    bool last_lmb = false, last_mmb = false, last_rmb = false;

    void on_update() {
        last_lmb = lmb;
        last_mmb = mmb;
        last_rmb = rmb;

        GLint nx, ny;
        glfwGetFramebufferSize(window, &nx, &ny);
        GLdouble _x, _y;
        glfwGetCursorPos(window, &_x, &_y);
        x = 0.5f + (float)_x;
        y = ny - 0.5f - (float)_y;
        lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
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

    virtual void on_update() = 0;
    virtual void on_draw() const = 0;
};


struct Widget : IWidget {
    bool hovered = false;
    bool selected = false;

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

    virtual void on_hover_enter() {}
    virtual void on_hover_leave() {}

    virtual void on_lmb_down() {}
    virtual void on_lmb_up() {}

    virtual void on_mmb_down() {}
    virtual void on_mmb_up() {}

    virtual void on_rmb_down() {}
    virtual void on_rmb_up() {}

    void on_update() override {
        auto raii = cur.translate(-position.x, -position.y);
        auto bbox = get_bounding_box();

        auto new_hovered = bbox.contains(cur.x, cur.y);
        if (!hovered && new_hovered) {
            on_hover_enter();
        } else if (hovered && !new_hovered) {
            on_hover_leave();
        }
        hovered = new_hovered;

        if (!cur.last_lmb && cur.lmb) {
            on_lmb_down();
        } else if (!cur.last_lmb && cur.lmb) {
            on_lmb_up();
        }

        if (!cur.last_mmb && cur.mmb) {
            on_mmb_down();
        } else if (!cur.last_mmb && cur.mmb) {
            on_mmb_up();
        }

        if (!cur.last_rmb && cur.rmb) {
            on_rmb_down();
        } else if (!cur.last_rmb && cur.rmb) {
            on_rmb_up();
        }

        update();
        for (auto const &child: children) {
            child->on_update();
        }
    }

    void on_draw() const override {
        glPushMatrix();
        glTranslatef(position.x, position.y, 0.f);
        draw();
        for (auto const &child: children) {
            child->on_draw();
        }
        glPopMatrix();
    }

    virtual void update() const {}
    virtual void draw() const {}
};


struct RectItem : Widget {
    AABB bbox;

    RectItem(AABB bbox)
        : bbox(bbox) {}

    AABB get_bounding_box() const override {
        return bbox;
    }

    void draw() const override {
        if (selected) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else if (hovered) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);
    }
};


struct Button : RectItem {
    std::string text;

    Button(AABB bbox, std::string text)
        : RectItem(bbox), text(text) {
    }

    AABB get_bounding_box() const override {
        return bbox;
    }

    void draw() const override {
        RectItem::draw();

        Font font("LiberationMono-Regular.ttf");
        //Font font("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc");
        font.set_font_size(30.f);
        font.set_fixed_width(bbox.nx);
        font.set_fixed_height(bbox.ny);
        glColor3f(1.f, 1.f, 1.f);
        font.render(bbox.x0, bbox.y0, text);
    }
};


struct MyWindow : Widget {
    MyWindow() {
        add_child<Button>(AABB(100, 100, 150, 50), "OK");
        add_child<Button>(AABB(300, 100, 150, 50), "Cancel");
        add_child<RectItem>(AABB(100, 300, 150, 50));
        add_child<RectItem>(AABB(300, 300, 150, 50));
    }

    AABB get_bounding_box() const override {
        return {0, 0, 500, 400};
    }
} win;


/*struct DemoWindow : Widget {
    DemoWindow() {
        add_child<MyWindow>();
    }
};*/


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
    win.on_update();
}


void draw_graphics() {
    glClearColor(0.2f, 0.3f, 0.5f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    win.on_draw();
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
