#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <FTGL/ftgl.h>
#include <memory>
#include <string>
#include <tuple>


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


struct CursorState {
    float x, y;
    bool lmb, mmb, rmb;
    void *lmb_on = nullptr;

    void on_update() {
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
} cur;


struct IWidget {
    IWidget() = default;
    IWidget(IWidget const &) = delete;
    IWidget &operator=(IWidget const &) = delete;
    virtual ~IWidget() = default;

    virtual void on_update() {}
    virtual void on_draw() const {}
};


struct Widget : IWidget {
    bool hovered = false;
    bool pressed = false;

    Widget *parent = nullptr;
    std::vector<std::shared_ptr<Widget>> children;

    virtual AABB get_bounding_box() const = 0;

    void on_update() override {
        auto bbox = get_bounding_box();
        hovered = bbox.contains(cur.x, cur.y);

        if (hovered && cur.lmb && !cur.lmb_on) {
            cur.lmb_on = this;
            pressed = true;
        }
        if (!cur.lmb) {
            cur.lmb_on = nullptr;
            pressed = false;
        }
    }
};


struct Button : Widget {
    AABB bbox;
    std::string text;

    Button(AABB bbox, std::string text)
        : bbox(bbox), text(text) {}

    AABB get_bounding_box() const override {
        return bbox;
    }

    void on_update() override {
        Widget::on_update();
    }

    void on_draw() const override {
        if (pressed) {
            glColor3f(0.375f, 0.5f, 1.0f);
        } else if (hovered) {
            glColor3f(0.75f, 0.5f, 0.375f);
        } else {
            glColor3f(0.375f, 0.375f, 0.375f);
        }
        glRectf(bbox.x0, bbox.y0, bbox.x0 + bbox.nx, bbox.y0 + bbox.ny);

        Font font("LiberationMono-Regular.ttf");
        //Font font("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc");
        font.set_font_size(30.f);
        font.set_fixed_width(bbox.nx);
        font.set_fixed_height(bbox.ny);

        glColor3f(1.f, 1.f, 1.f);
        font.render(bbox.x0, bbox.y0, text);
    }
};


Button btn1({100, 100, 150, 50}, "OK");
Button btn2({300, 100, 150, 50}, "Cancel");


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

    btn1.on_update();
    btn2.on_update();
}


void draw_graphics() {
    glClearColor(0.2f, 0.3f, 0.5f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    btn1.on_draw();
    btn2.on_draw();
}


int main() {
    if (!glfwInit()) {
        const char *err = "unknown error"; glfwGetError(&err);
        fprintf(stderr, "Failed to initialize GLFW library: %s\n", err);
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
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
