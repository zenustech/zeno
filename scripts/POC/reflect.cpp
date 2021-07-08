#include <iostream>
#include <string_view>
using namespace std;

namespace refl {
    namespace details {
        template <class T>
        auto type_name() {
#if defined(__clang__)
          string_view name{
            __PRETTY_FUNCTION__ + 37,
            sizeof(__PRETTY_FUNCTION__) - 39};
#elif defined(__GNUC__)
          string_view name{
            __PRETTY_FUNCTION__ + 42,
            sizeof(__PRETTY_FUNCTION__) - 44};
#else
#error "Unsupported complier."
#endif
          return name;
        }
    }
}

int main() {
  cout << "[" << refl::details::type_name<std::string>() << "]" << endl;
  return 0;
}
