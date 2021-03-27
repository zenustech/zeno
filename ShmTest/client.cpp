#include <Hg/IPC/Socket.hpp>


int main(int argc, char** argv)
{
  Socket sock("/tmp/UNIX.domain");
  char buf[] = "message from client";
  sock.write(buf, sizeof(buf));
  return 0;
}
