#pragma once


#include <memory>
#include <zs/ztd/zany.h>
#include <zs/editor/GL/Camera.h>


namespace zs::editor::GL {


struct VisRender {
    virtual void render(GL::Camera *camera) = 0;
    virtual ~VisRender() = default;

    static std::unique_ptr<VisRender> make_for(ztd::zany object);
};


}
