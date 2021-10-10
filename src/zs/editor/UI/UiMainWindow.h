#pragma once


#include <zs/editor/UI/UiDopScene.h>
#include <zs/editor/UI/UiVisViewport.h>
#include <zs/editor/UI/TabPanel.h>


namespace zs::editor::UI {


struct UiMainWindow : Widget {
    TabPanel<UiDopScene> *scenes;
    UiVisViewport *viewport;

    UiMainWindow() {
        scenes = add_child<TabPanel<UiDopScene>>();
        auto scene1 = scenes->add_element();
        auto scene2 = scenes->add_element();
        scenes->show_element(scene2);
        viewport = add_child<UiVisViewport>();
        viewport->bbox = {0, 0, 1600, 460};
        viewport->position = {0, 440};
    }

    std::vector<ztd::zany> view_results() const {
        std::vector<ztd::zany> res;
        for (auto *scene: scenes->elements) {
            if (auto obj = scene->view_result; obj.has_value())
                res.push_back(obj);
        }
        return res;
    }
};


}
