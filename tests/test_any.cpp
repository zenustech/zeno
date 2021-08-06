#include <catch2/catch.hpp>
#include <zeno/zeno.h>
#include <zeno/utils/any.h>

TEST_CASE("cast any of int to float", "[any]") {
    int i = 42;
    zeno::any a = i;
    float f = zeno::smart_any_cast<float>(a);
    REQUIRE(f == 42.0f);

    i = 32;
    a = i;
    f = zeno::smart_any_cast<float>(a);
    REQUIRE(f == 32.0f);
}

TEST_CASE("cast any of float to int", "[any]") {
    float f = 42.8f;
    zeno::any a = f;
    int i = zeno::smart_any_cast<int>(a);
    REQUIRE(i == 42);

    f = 32.2f;
    a = f;
    i = zeno::smart_any_cast<int>(a);
    REQUIRE(i == 32);
}

TEST_CASE("cast any of vec3i to vec3f", "[any]") {
    zeno::vec3i i(42, 985, 211);
    zeno::any a = i;
    zeno::vec3f f = zeno::smart_any_cast<zeno::vec3f>(a);
    REQUIRE(allTrue(f == zeno::vec3f(42.0f, 985.0f, 211.0f)));
}

TEST_CASE("cast any of vec3f to vec3i", "[any]") {
    zeno::vec3f f(42.0f, 985.99f, 211.3f);
    zeno::any a = f;
    zeno::vec3i i = zeno::smart_any_cast<zeno::vec3i>(a);
    REQUIRE(allTrue(i == zeno::vec3i(42, 985, 211)));
}
