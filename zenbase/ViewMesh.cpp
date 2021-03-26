#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <Hg/IPC/shm.hpp>

namespace zenvis {

struct ViewMesh : zen::INode {
  std::vector<float> vertex_data;

  virtual void apply() override {
    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();

    size_t memsize = mesh->vertices.size() * 8 * sizeof(float);
    SHM shm("/tmp/mem", memsize);

    int memi = 0;
    float *memdata = (float *)shm.data();

    for (int i = 0; i < mesh->vertices.size(); i++) {
      memdata[memi++] = mesh->vertices[i].x;
      memdata[memi++] = mesh->vertices[i].y;
      memdata[memi++] = mesh->vertices[i].z;
      memdata[memi++] = mesh->uvs[i].x;
      memdata[memi++] = mesh->uvs[i].y;
      memdata[memi++] = mesh->normals[i].x;
      memdata[memi++] = mesh->normals[i].y;
      memdata[memi++] = mesh->normals[i].z;
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
