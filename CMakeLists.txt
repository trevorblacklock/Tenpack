cmake_minimum_required(VERSION 3.15)

project(tenpack CXX)

find_package(benchmark REQUIRED)
find_package(hwy REQUIRED)
find_package(fmt REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wpedantic -Wabi -Wshadow")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

file(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    benchmark::benchmark
    hwy::hwy
    fmt::fmt)
