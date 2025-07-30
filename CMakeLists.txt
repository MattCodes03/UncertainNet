cmake_minimum_required(VERSION 3.15)
project(UncertainNet)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/x64-linux/share" ${CMAKE_PREFIX_PATH})
endif()

include(FetchContent)

# Disable autodiff extras
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Fetch autodiff
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(autodiff)

# Find Eigen3 via vcpkg
find_package(Eigen3 CONFIG REQUIRED)

# Fetch Catch2
FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
)
FetchContent_MakeAvailable(catch2)

# -----------------------------------------------------------------------------
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_library(UncertainNetLib ${LIB_SOURCES})

# Include all headers in include/
target_include_directories(UncertainNetLib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(UncertainNetLib PUBLIC Eigen3::Eigen autodiff::autodiff)

# -----------------------------------------------------------------------------
# Build all example executables from examples/ folder (each .cpp a separate exe)
file(GLOB EXAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/examples/*.cpp")
foreach(example_file IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name} PRIVATE UncertainNetLib)

    # Put example executables into build/examples/
    set_target_properties(${example_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
        FOLDER "examples"
    )
endforeach()

# -----------------------------------------------------------------------------
# Build all test executables from tests/ folder (each .cpp a separate exe)
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")
foreach(test_file IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE UncertainNetLib Catch2::Catch2)
    add_test(NAME ${test_name} COMMAND ${test_name})

    # Put test executables into build/tests/
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        FOLDER "tests"
    )
endforeach()

# Enable CTest testing
include(CTest)
