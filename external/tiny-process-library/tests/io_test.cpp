#include "process.hpp"
#include <cassert>
#include <iostream>

using namespace std;
using namespace TinyProcessLib;

int main() {
  auto output=make_shared<string>();
  auto error=make_shared<string>();
  {
    Process process("echo Test", "", [output](const char *bytes, size_t n) {
      *output+=string(bytes, n);
    });
    assert(process.get_exit_status()==0);
    assert(output->substr(0, 4)=="Test");
    output->clear();
  }
  
#ifndef _WIN32
  {
    Process process([] {
      cout << "Test" << endl;
      exit(0);
    }, [output](const char *bytes, size_t n) {
      *output+=string(bytes, n);
    });
    assert(process.get_exit_status()==0);
    assert(output->substr(0, 4)=="Test");
    output->clear();
  }
#endif
  
  {
    Process process("echo Test && ls an_incorrect_path", "", [output](const char *bytes, size_t n) {
      *output+=string(bytes, n);
    }, [error](const char *bytes, size_t n) {
      *error+=string(bytes, n);
    });
    assert(process.get_exit_status()>0);
    assert(output->substr(0, 4)=="Test");
    assert(!error->empty());
    output->clear();
    error->clear();
  }
  
  {
    Process process("bash", "", [output](const char *bytes, size_t n) {
      *output+=string(bytes, n);
    }, nullptr, true);
    process.write("echo Test\n");
    process.write("exit\n");
    assert(process.get_exit_status()==0);
    assert(output->substr(0, 4)=="Test");
    output->clear();
  }
  
  {
    Process process("cat", "", [output](const char *bytes, size_t n) {
      *output+=string(bytes, n);
    }, nullptr, true);
    process.write("Test\n");
    process.close_stdin();
    assert(process.get_exit_status()==0);
    assert(output->substr(0, 4)=="Test");
    output->clear();
  }
  
  {
    Process process("sleep 5");
    int exit_status=-2;
    assert(!process.try_get_exit_status(exit_status));
    assert(exit_status==-2);
    this_thread::sleep_for(chrono::seconds(3));
    assert(!process.try_get_exit_status(exit_status));
    assert(exit_status==-2);
    this_thread::sleep_for(chrono::seconds(5));
    assert(process.try_get_exit_status(exit_status));
    assert(exit_status==0);
  }
}
