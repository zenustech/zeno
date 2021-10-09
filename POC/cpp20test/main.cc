#include <array>
#include "hello.hh"
#include "world.hh"

int main() {
    hello();
    world();
    auto p = get_array<int>();
}
