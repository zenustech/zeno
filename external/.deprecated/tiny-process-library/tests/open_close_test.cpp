#include "process.hpp"
#include <iostream>

using namespace std;
using namespace TinyProcessLib;

int main() {
  bool stdout_error=false;
  for(size_t c=0;c<10000;c++) {
    Process process("echo Hello World "+to_string(c), "", [&stdout_error, c](const char *bytes, size_t n) {
      if(string(bytes, n)!="Hello World "+to_string(c)+"\n")
        stdout_error=true;
    }, [](const char *, size_t) {
    }, true);
    auto exit_status=process.get_exit_status();
    if(exit_status!=0) {
      cerr << "Process returned failure." << endl;
      return 1;
    }
    if(stdout_error) {
      cerr << "Wrong output to stdout." << endl;
      return 1;
    }
  }
 
  return 0;
}