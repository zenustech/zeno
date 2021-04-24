#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>
#include <cstring>
#include <vector>

namespace zenbase {

struct ViewParticles : zen::INode {
  std::vector<char> shader;

  std::vector<char> get_memory_data() {
    auto pars = get_input("pars")->as<zenbase::ParticlesObject>();
    size_t vertex_count = pars->pos.size();

    std::vector<char> memory(vertex_count * 6 * sizeof(float));

    size_t memi = 0;
    float *fdata = (float *)memory.data();
    for (int i = 0; i < vertex_count; i++) {
      fdata[memi++] = pars->pos[i].x;
      fdata[memi++] = pars->pos[i].y;
      fdata[memi++] = pars->pos[i].z;
      fdata[memi++] = pars->vel[i].x;
      fdata[memi++] = pars->vel[i].y;
      fdata[memi++] = pars->vel[i].z;
    }

    return memory;
  }

  std::string get_data_type() const {
    return "PARS";
  }

  virtual void apply() override {
    Socket sock("/tmp/zenipc/command");

    auto memory = get_memory_data();
    SharedMemory shm_memory("/tmp/zenipc/memory", memory.size());
    std::memcpy(shm_memory.data(), memory.data(), memory.size());
    shm_memory.release();

    SharedMemory shm_shader("/tmp/zenipc/shader", shader.size());
    std::memcpy(shm_shader.data(), shader.data(), shader.size());
    shm_shader.release();

    dprintf(sock.filedesc(), "@%s %zd %zd\n",
        get_data_type().c_str(), memory.size(), shader.size());
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
