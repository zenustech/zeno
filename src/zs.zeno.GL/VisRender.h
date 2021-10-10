#pragma once


#include <memory>
#include <zeno2/ztd/zany.h>
#include <zeno2/GL/Camera.h>


namespace zeno2::GL {


struct VisRender {
    virtual void render(GL::Camera *camera) = 0;
    virtual ~VisRender() = default;

    static std::unique_ptr<VisRender> make_for(ztd::zany object);
};


}
