#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>
#include <cstring>
#include <vector>

namespace zenbase {

struct ViewParticles : zen::INode {
  virtual void apply() override {

    /**************/
    auto pars = get_input("pars")->as<zenbase::ParticlesObject>();
    size_t vertex_count = pars->pos.size();
    size_t memsize = vertex_count * 6 * sizeof(float);
    /**************/

    Socket sock("/tmp/zenipc/command");
    SharedMemory shm("/tmp/zenipc/memory", memsize);
    float *memdata = (float *)shm.data();

    /**************/
    int memi = 0;
    for (int i = 0; i < vertex_count; i++) {
      memdata[memi++] = pars->pos[i].x;
      memdata[memi++] = pars->pos[i].y;
      memdata[memi++] = pars->pos[i].z;
      memdata[memi++] = pars->vel[i].x;
      memdata[memi++] = pars->vel[i].y;
      memdata[memi++] = pars->vel[i].z;
    }
    /**************/

    shm.release();
    dprintf(sock.filedesc(), "@PARS %zd\n", memsize);
    sock.readchar();  // wait server to be ready
  }
};

static int defViewParticles = zen::defNodeClass<ViewParticles>("ViewParticles",
    { /* inputs: */ {
        "pars",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
