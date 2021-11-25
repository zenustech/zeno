#include <FL/Fl.H>
#include <FL/fl_draw.H>
#include <FL/Fl_Double_Window.H>
#include <vector>

class ZNode {
public:
    ZNode() = default;

    void draw() {
        fl_color(FL_WHITE);
        fl_rectf(x(), y(), w(), h());
    }
};

class ZScene : public Fl_Widget {
    std::vector<ZNode> nodes;

public:
    ZScene(int X, int Y, int W, int H, const char *L = nullptr)
        : Fl_Widget(X, Y, W, H, L) {
        color(FL_WHITE);
    }

    void draw() {
        fl_color(FL_WHITE);
        fl_rectf(x(), y(), w(), h());
        fl_color(FL_BLACK);
        int x1 = x(),       y1 = y();
        int x2 = x()+w()-1, y2 = y()+h()-1;
        fl_line(x1, y1, x2, y2);
        fl_line(x1, y2, x2, y1);
    }
};

int main() {
    Fl_Double_Window win(200, 200, "Zeno Editor");
    ZScene scene(10, 10, win.w() - 20, win.h() - 20);
    win.resizable(scene);
    win.show();
    return Fl::run();
}
