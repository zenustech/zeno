#include <catch2/catch.hpp>

TEST_CASE("numeric operators", "[numeric]") {
    int a = 1;
    REQUIRE(a + 1 == 2);
}
