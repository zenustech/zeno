#include <zen/zen.h>
#include <zen/MeshObject.h>

namespace zenvis {

struct ViewMesh : zen::INode {
  size_t vertex_count;
  std::vector<float> vertex_data;

  virtual void apply() override {
    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();

    vertex_data.clear();
    vertex_count = mesh->vertices.size();
    for (int i = 0; i < vertex_count; i++) {
      vertex_data.push_back(mesh->vertices[i].x);
      vertex_data.push_back(mesh->vertices[i].y);
      vertex_data.push_back(mesh->vertices[i].z);
      vertex_data.push_back(mesh->uvs[i].x);
      vertex_data.push_back(mesh->uvs[i].y);
      vertex_data.push_back(mesh->normals[i].x);
      vertex_data.push_back(mesh->normals[i].y);
      vertex_data.push_back(mesh->normals[i].z);
    }

    for (int i = 0; i < vertex_data.size(); i++) {
      printf("%f\n", vertex_data[i]);
    }
  }
};

static int defViewMesh = zen::defNodeClass<ViewMesh>("ViewMesh",
    { /* inputs: */ {
        "mesh",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
