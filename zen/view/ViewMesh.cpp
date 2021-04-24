#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <Hg/StrUtils.h>
#include <cstring>
#include <vector>
#include "ViewNode.h"

namespace zenbase {

struct ViewMesh : ViewNode {
  virtual std::vector<char> get_shader() override {
    return std::vector<char>(0);
  }

  virtual std::vector<char> get_memory() override {
    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();
    size_t vertex_count = mesh->vertices.size();
    std::vector<char> memory(vertex_count * 8 * sizeof(float));

    size_t memi = 0;
    float *fdata = (float *)memory.data();
    for (int i = 0; i < vertex_count; i++) {
      fdata[memi++] = mesh->vertices[i].x;
      fdata[memi++] = mesh->vertices[i].y;
      fdata[memi++] = mesh->vertices[i].z;
      fdata[memi++] = mesh->uvs[i].x;
      fdata[memi++] = mesh->uvs[i].y;
      fdata[memi++] = mesh->normals[i].x;
      fdata[memi++] = mesh->normals[i].y;
      fdata[memi++] = mesh->normals[i].z;
    }
    return memory;
  }

  virtual std::string get_data_type() const override {
    return "MESH";
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
