#pragma once


#include <z2/UI/UiDopScene.h>
#include <z2/UI/UiVisViewport.h>


namespace z2::UI {


struct UiMainWindow : Widget {
    UiDopScene *scene;
    UiVisViewport *viewport;

    UiMainWindow() {
        scene = add_child<UiDopScene>();
        viewport = add_child<UiVisViewport>();
        viewport->bbox = {0, 0, 1600, 460};
        viewport->position = {0, 440};
    }

    std::vector<std::any> view_results() const {
        std::vector<std::any> res;
        if (auto obj = scene->view_result; obj.has_value())
            res.push_back(obj);
        return res;
    }
};


}
