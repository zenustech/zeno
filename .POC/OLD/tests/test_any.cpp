#include <catch2/catch.hpp>
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/Any.h>

TEST_CASE("cast any of int to float", "[any]") {
    int i = 42;
    zeno::Any a = i;
    float f = zeno::smart_any_cast<float>(a);
    REQUIRE(f == 42.0f);

    i = 32;
    a = i;
    f = zeno::smart_any_cast<float>(a);
    REQUIRE(f == 32.0f);
}

TEST_CASE("cast any of float to int", "[any]") {
    float f = 42.8f;
    zeno::Any a = f;
    int i = zeno::smart_any_cast<int>(a);
    REQUIRE(i == 42);

    f = 32.2f;
    a = f;
    i = zeno::smart_any_cast<int>(a);
    REQUIRE(i == 32);
}

TEST_CASE("cast any of vec3i to vec3f", "[any]") {
    zeno::vec3i i(42, 985, 211);
    zeno::Any a = i;
    zeno::vec3f f = zeno::smart_any_cast<zeno::vec3f>(a);
    REQUIRE(alltrue(f == zeno::vec3f(42.0f, 985.0f, 211.0f)));
}

TEST_CASE("cast any of vec3f to vec3i", "[any]") {
    zeno::vec3f f(42.0f, 985.99f, 211.3f);
    zeno::Any a = f;
    zeno::vec3i i = zeno::smart_any_cast<zeno::vec3i>(a);
    REQUIRE(alltrue(i == zeno::vec3i(42, 985, 211)));
}

TEST_CASE("static assertion of underlying type", "[any]") {
    REQUIRE(std::is_same_v<
            zeno::any_underlying_type_t<float>,
            zeno::scalar_type_variant>);
    REQUIRE(std::is_same_v<
            zeno::any_underlying_type_t<std::shared_ptr<zeno::PrimitiveObject>>,
            std::shared_ptr<zeno::IObject>>);
    REQUIRE(std::is_same_v<
            zeno::any_underlying_type_t<zeno::vec4i>,
            zeno::vector_type_variant<4>>);
}

// TODO: add shared_ptr tests too
