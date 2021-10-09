#pragma ocne


#include <memory>
#include <z2/ztd/zany.h>
#include <z2/GL/Camera.h>


namespace z2::GL {


struct VisRender {
    virtual void render(GL::Camera *camera) = 0;
    virtual ~VisRender() = default;

    static std::unique_ptr<VisRender> make_for(ztd::zany object);
};


}
