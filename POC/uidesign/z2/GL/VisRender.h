#pragma ocne


#include <any>
#include <memory>
#include <z2/GL/Camera.h>
#include <z2/ds/Mesh.h>


namespace z2::GL {


struct VisRender {
    virtual void render(GL::Camera *camera) = 0;
    virtual ~VisRender() = default;

    static std::unique_ptr<VisRender> make_for(std::any object);
};


struct VisRender_Mesh : VisRender {
    std::shared_ptr<ds::Mesh> mesh;
    VisRender_Mesh(std::shared_ptr<ds::Mesh> mesh) : mesh(mesh) {}

    static GL::Program *_get_shader();
    void render(GL::Camera *camera) override;
};


}
