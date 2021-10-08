#pragma once


#include <zeno2/UI/UiDopScene.h>
#include <zeno2/UI/UiVisViewport.h>


namespace zeno2::UI {


struct UiMainWindow : Widget {
    UiDopScene *scene;
    UiVisViewport *viewport;

    UiMainWindow() {
        scene = add_child<UiDopScene>();
        viewport = add_child<UiVisViewport>();
        viewport->bbox = {0, 0, 1600, 460};
        viewport->position = {0, 440};
    }

    std::vector<ztd::zany> view_results() const {
        std::vector<ztd::zany> res;
        if (auto obj = scene->view_result; obj.has_value())
            res.push_back(obj);
        return res;
    }
};


}
