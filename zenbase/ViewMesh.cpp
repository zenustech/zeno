#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>
#include <cstring>
#include <vector>

namespace zenvis {

class ViewCommand {
  std::vector<char> m;

public:
  ViewCommand(const char *type, const char *path, size_t size) {
    m.resize(sizeof(size_t) + strlen(type) + 1 + strlen(path) + 1);
    std::memcpy(m.data(), &size, sizeof(size_t));
    std::strcpy(m.data() + sizeof(size_t), type);
    std::strcat(m.data() + sizeof(size_t), path);
  }

  void *data() {
    return m.data();
  }

  size_t size() {
    return m.size();
  }
};

struct ViewMesh : zen::INode {
  std::vector<float> vertex_data;

  virtual void apply() override {
    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();

    size_t memsize = mesh->vertices.size() * 8 * sizeof(float);

    const char *path = "/tmp/zenipc/mesh01";
    SharedMemory shm(path, memsize);

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

    shm.release();

    Socket sock("/tmp/zenipc/command", true);
    ViewCommand cmd("MESH", path, memsize);
    sock.write(cmd.data(), cmd.size());
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
