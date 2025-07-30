// tests/test_basic.cpp

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

TEST_CASE("Basic arithmetic test")
{
    REQUIRE(1 + 1 == 2);
    REQUIRE(2 * 2 == 4);
}
