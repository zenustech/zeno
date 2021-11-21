#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
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

  explicit Socket(int conn) : conn(conn) {}

public:
  Socket() = default;  // for listen(Socket *)

  int filedesc() const {
    return conn;
  }

  void set_nonblock(bool nonblock = true) const {
    int flag = ::fcntl(conn, F_GETFL);
    if (nonblock)
      flag |= O_NONBLOCK;
    else
      flag &= ~O_NONBLOCK;
    ::fcntl(conn, F_SETFL, flag);
  }

  void close() {
    if (conn >= 0) {
      ::close(conn);
      conn = -1;
    }
  }

  void write(const void *buf, size_t size) const {
    int ret = ::write(conn, buf, size);
    if (ret < 0) {
      ::perror("write");
      return;
    }
  }

  size_t read(void *buf, size_t size) const {
    ssize_t ret = ::read(conn, buf, size);
    if (ret < 0) {
      if (errno != EAGAIN)
        ::perror("read");
      return 0;
    }
    return ret;
  }

  void writechar(char c) const {
    char buf[1] = {c};
    this->write(buf, 1);
  }

  char readchar() const {
    char buf[1];
    this->read(buf, 1);
    return buf[0];
  }

  explicit Socket(const char *domain, bool streamed = true) {
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
    int filedesc() const {
      return conn;
    }

    void set_nonblock(bool nonblock = true) const {
      int flag = ::fcntl(conn, F_GETFL);
      if (nonblock)
        flag |= O_NONBLOCK;
      else
        flag &= ~O_NONBLOCK;
      ::fcntl(conn, F_SETFL, flag);
    }

    bool listen(Socket *sock, int backlog = 10) const {
      int ret = ::listen(conn, backlog);
      if (ret < 0) {
        ::perror("listen");
        return false;
      }

      struct sockaddr_un clt_addr;
      socklen_t len = sizeof(clt_addr);
      int fd = ::accept(conn, (struct sockaddr *)&clt_addr, &len);
      if (fd < 0) {
        if (errno != EAGAIN)
          ::perror("accept");
        return false;
      }
      sock->conn = fd;
      return true;
    }

    explicit Server(const char *domain, bool streamed = true) {
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
