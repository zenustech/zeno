#include <sys/mman.h>  
#include <sys/types.h>  
#include <sys/stat.h>  
#include <fcntl.h>  
#include <unistd.h>  
#include <stdio.h>  


struct SHM {
  void *p{nullptr};
  size_t size;

  SHM(size_t size_, const char *path)
    : size(size_)
  {
    int fd = open(path, O_RDWR, 00777);
    p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
  }

  ~SHM()
  {
    munmap(p, size);
  }
};


int main(int argc, char** argv)  
{  
  SHM shm(32, "/tmp/mem");
  ((int*)shm.p)[0] = 1;
  return 0;  
}  
