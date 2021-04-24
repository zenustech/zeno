#pragma once

#include <zen/zen.h>
#include <Hg/IPC/SharedMemory.hpp>
#include <Hg/IPC/Socket.hpp>

namespace zenbase {

struct ViewNode : zen::INode {
  virtual std::vector<char> get_shader() = 0;
  virtual std::vector<char> get_memory() = 0;
  virtual std::string get_data_type() const = 0;

  virtual void apply() override {
    Socket sock("/tmp/zenipc/command");

    auto memory = get_memory();
    SharedMemory shm_memory("/tmp/zenipc/memory", memory.size());
    std::memcpy(shm_memory.data(), memory.data(), memory.size());
    shm_memory.release();

    auto shader = get_shader();
    SharedMemory shm_shader("/tmp/zenipc/shader", shader.size());
    std::memcpy(shm_shader.data(), shader.data(), shader.size());
    shm_shader.release();

    dprintf(sock.filedesc(), "@%s %zd %zd\n",
        get_data_type().c_str(), memory.size(), shader.size());
    sock.readchar();
  }
};
}
