#include "process.hpp"
#include <iostream>
#include <vector>
#include <atomic>

using namespace std;
using namespace TinyProcessLib;

int main() {
  atomic<bool> stdout_error(false);
  atomic<bool> exit_status_error(false);
  vector<thread> threads;
  for(size_t ct=0;ct<4;ct++) {
    threads.emplace_back([&stdout_error, &exit_status_error, ct]() {
      for(size_t c=0;c<2500;c++) {
        Process process("echo Hello World "+to_string(c)+" "+to_string(ct), "", [&stdout_error, ct, c](const char *bytes, size_t n) {
          if(string(bytes, n)!="Hello World "+to_string(c)+" "+to_string(ct)+"\n")
            stdout_error=true;
        }, [](const char *, size_t) {
        }, true);
        auto exit_status=process.get_exit_status();
        if(exit_status!=0)
          exit_status_error=true;
        if(exit_status_error)
          return;
        if(stdout_error)
          return;
      }
    });
  }
  
  for(auto &thread: threads)
    thread.join();
  
  if(stdout_error) {
    cerr << "Wrong output to stdout." << endl;
    return 1;
  }
  if(exit_status_error) {
    cerr << "Process returned failure." << endl;
    return 1;
  }
  
  return 0;
}