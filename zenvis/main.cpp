#include <Hg/IPC/Socket.hpp>


int main(int argc, char **argv)
{
  Socket::Server serv("/tmp/zenipc/command");

  while (true) {
    Socket sock = serv.listen();

    char buf[1024];
    size_t num = sock.read(buf, sizeof(buf));
    buf[num] = 0;

    char type[32] = {0};
    char mempath[1024] = {0};
    size_t vertex_count;
    sscanf(buf, "@%s %zd %s", type, &vertex_count, mempath);
    printf("recv: %s, %zd, %s\n", type, vertex_count, mempath);
  }
 
  return 0;
}
