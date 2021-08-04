#include <functional>
#include <typeinfo>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <any>
#include "Method.h"


int main() {
    auto method = zfp::make_method([=] (int val) -> int {
        return -val;
    });
    int ret = std::any_cast<int>(method(1));
    std::cout << ret << std::endl;
}
