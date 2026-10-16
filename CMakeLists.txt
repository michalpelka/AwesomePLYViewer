cmake_minimum_required(VERSION 3.16)
project(splat_viewer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(raylib 5.5 REQUIRED)

# happly: header-only PLY reader/writer (MIT), https://github.com/nmwsharp/happly
# The repo has no CMakeLists.txt, so FetchContent just downloads it and we add its folder to the include path.
include(FetchContent)
FetchContent_Declare(happly
    GIT_REPOSITORY https://github.com/nmwsharp/happly.git
    GIT_TAG        ed5480454d15e5287370eadc3682e9d40ba395e1)
FetchContent_MakeAvailable(happly)

add_executable(splat_viewer splat_viewer.cpp)
target_include_directories(splat_viewer PRIVATE ${happly_SOURCE_DIR})
target_link_libraries(splat_viewer PRIVATE raylib m)

# The installed raylib is a static library, so its own dependencies have to be linked here.
if(UNIX AND NOT APPLE)
    find_package(OpenGL REQUIRED)
    find_package(X11 REQUIRED)
    target_link_libraries(splat_viewer PRIVATE OpenGL::GL ${X11_LIBRARIES} pthread dl rt)
endif()
