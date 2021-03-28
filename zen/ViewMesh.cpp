#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>
#include <cstring>
#include <vector>

namespace zenbase {

struct ViewMesh : zen::INode {
  virtual void apply() override {

    /**************/
    auto mesh = get_input("mesh")->as<zenbase::MeshObject>();
    size_t vertex_count = mesh->vertices.size();
    size_t memsize = vertex_count * 8 * sizeof(float);
    /**************/

    Socket sock("/tmp/zenipc/command");
    SharedMemory shm("/tmp/zenipc/memory", memsize);
    float *memdata = (float *)shm.data();

    /**************/
    int memi = 0;
    for (int i = 0; i < vertex_count; i++) {
      memdata[memi++] = mesh->vertices[i].x;
      memdata[memi++] = mesh->vertices[i].y;
      memdata[memi++] = mesh->vertices[i].z;
      memdata[memi++] = mesh->uvs[i].x;
      memdata[memi++] = mesh->uvs[i].y;
      memdata[memi++] = mesh->normals[i].x;
      memdata[memi++] = mesh->normals[i].y;
      memdata[memi++] = mesh->normals[i].z;
    }
    /**************/

    shm.release();
    dprintf(sock.filedesc(), "@MESH %zd\n", memsize);
    sock.readchar();  // wait server to be ready
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
