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

# Optional example scenes (~200 MB), downloaded into examples/ at configure time:
#     cmake -S . -B build -DSPLAT_VIEWER_EXAMPLES=ON
# Files that already exist are skipped. URLs are pinned to a repo revision and every download is
# checked against its SHA-256. Where the scenes come from and their licenses: see README.md.
option(SPLAT_VIEWER_EXAMPLES "Download example splat scenes into examples/" OFF)

function(download_example name url sha256)
    set(dest "${CMAKE_CURRENT_SOURCE_DIR}/examples/${name}")
    if(EXISTS "${dest}")
        return()
    endif()
    message(STATUS "Downloading example ${name}")
    # Download to a temporary name first, so an interrupted download is not mistaken for a finished one.
    file(DOWNLOAD "${url}" "${dest}.part" SHOW_PROGRESS STATUS status)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        list(GET status 1 reason)
        file(REMOVE "${dest}.part")
        message(WARNING "Could not download example ${name}: ${reason}")
        return()
    endif()
    file(SHA256 "${dest}.part" actual)
    if(NOT actual STREQUAL sha256)
        file(REMOVE "${dest}.part")
        message(WARNING "Example ${name} has an unexpected SHA-256 (${actual}); the file was removed")
        return()
    endif()
    file(RENAME "${dest}.part" "${dest}")
endfunction()

if(SPLAT_VIEWER_EXAMPLES)
    set(hf_splat_data "https://huggingface.co/cakewalk/splat-data/resolve/2c9796f59101d77d23f6eee53baa5018c9a6fcdc")
    set(hf_voxel51 "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/6aaba1b1dd0439c0db464842f8a6600a37e0eae9")
    download_example(nike/nike.splat "${hf_splat_data}/nike.splat"
        35888459aeddcda0b75a2bd76bba5a94c6e32aec0a0d566e96cd10dfd6e18b3c)    # 9 MB
    download_example(plush/plush.splat "${hf_splat_data}/plush.splat"
        80e0ae670ee980f166d9b00045d33710dd77e65ff004f1886ab442220e2c48fa)    # 9 MB
    download_example(train/train.ply "${hf_voxel51}/FO_dataset/train/point_cloud/iteration_7000/point_cloud.ply"
        9f5c643fa18cb5ed6a11441f073133ed1034e3b0656bce5b7b5db3db49161366)    # 184 MB
endif()
