#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>


int main(int argc, char** argv)  
{  
  int sock = socket(PF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) {
    perror("socket");
    return 1;
  }

  const char *domain = "/tmp/UNIX.domain";

  struct sockaddr_un srv_addr;
  srv_addr.sun_family = AF_UNIX;
  strcpy(srv_addr.sun_path, domain);
  unlink(domain);

  int ret = bind(sock, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
  if (ret < 0) {
    perror(domain);
    close(sock);
    unlink(domain);
    return 1;
  }

  ret = listen(sock, 1);
  if (ret < 0) {
    perror(domain);
    close(sock);
    unlink(domain);
    return 1;
  }

  struct sockaddr_un clt_addr;
  socklen_t len = sizeof(clt_addr);
  int conn = accept(sock, (struct sockaddr *)&clt_addr, &len);
  if (conn < 0) {
    perror("accept");
    close(sock);
    unlink(domain);
    return 1;
  }

  char buf[1024];

  int num = read(conn, buf, sizeof(buf));
  printf("recv: %d %s\n", num, buf);

  close(conn);
  close(sock);
  unlink(domain);
  return 0;  
}  
