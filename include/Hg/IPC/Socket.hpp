#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <memory>


class Socket {
  int conn{-1};

  void create(bool streamed) {
    conn = ::socket(PF_UNIX, streamed ? SOCK_STREAM : SOCK_DGRAM, 0);
    if (conn < 0) {
      ::perror("socket");
      return;
    }
  }

  void open(const char *domain) {
    struct sockaddr_un srv_addr;
    srv_addr.sun_family = AF_UNIX;
    ::strcpy(srv_addr.sun_path, domain);

    int ret = ::connect(conn, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
    if (ret < 0) {
      ::perror(domain);
      return;
    }
  }

  explicit Socket(int conn = -1) : conn(conn) {}

public:
  void close() {
    if (conn >= 0) {
      ::close(conn);
      conn = -1;
    }
  }

  void write(const void *buf, size_t size) {
    int ret = ::write(conn, buf, size);
    if (ret < 0) {
      perror("write");
      return;
    }
  }

  size_t read(void *buf, size_t size) {
    ssize_t ret = ::read(conn, buf, size);
    if (ret < 0) {
      perror("read");
      return 0;
    }
    return ret;
  }

  Socket(const char *domain, bool streamed) {
    this->create(streamed);
    this->open(domain);
  }

  Socket(Socket const &) = delete;
  Socket &operator=(Socket const &) = delete;

  ~Socket() {
    this->close();
  }

  class Server {
    int conn{-1};

    void create(bool streamed) {
      conn = ::socket(PF_UNIX, streamed ? SOCK_STREAM : SOCK_DGRAM, 0);
      if (conn < 0) {
        ::perror("socket");
        return;
      }
    }

    void close() {
      if (conn >= 0) {
        ::close(conn);
        conn = -1;
      }
    }

    void open(const char *domain) {
      struct sockaddr_un srv_addr;
      srv_addr.sun_family = AF_UNIX;
      ::strcpy(srv_addr.sun_path, domain);
      ::unlink(domain);

      int ret = ::bind(conn, (struct sockaddr *)&srv_addr, sizeof(srv_addr));
      if (ret < 0) {
        ::perror(domain);
        return;
      }
    }

  public:
    Socket listen(int backlog = 1) {
      int ret = ::listen(conn, backlog);
      if (ret < 0) {
        ::perror("listen");
        return Socket();
      }

      struct sockaddr_un clt_addr;
      socklen_t len = sizeof(clt_addr);
      int fd = ::accept(conn, (struct sockaddr *)&clt_addr, &len);
      if (fd < 0) {
        ::perror("accept");
        return Socket();
      }
      return Socket(fd);
    }

    Server(const char *domain, bool streamed) {
      this->create(streamed);
      this->open(domain);
    }

    Server(Server const &) = delete;
    Server &operator=(Server const &) = delete;

    ~Server() {
      this->close();
    }
  };
};
