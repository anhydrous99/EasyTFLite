cmake_minimum_required(VERSION 3.14)

option(BUILD_TESTS "Build the Tests" ON)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.204.tar.gz"
        SHA1 "32cfed254da901f6f184027d530d8da47e035b85"
)

project(EasyTFLite)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}/cmake/modules/")

hunter_add_package(Boost COMPONENTS filesystem)
hunter_add_package(Eigen)
hunter_add_package(glog)
hunter_add_package(GTest)
hunter_add_package(OpenCV)
hunter_add_package(ZLIB)

find_package(Boost CONFIG REQUIRED filesystem)
find_package(Eigen3 CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorFlowLite)
find_package(ZLIB CONFIG REQUIRED)

add_library(EasyTFLite src/TFLite.cpp)
target_link_libraries(EasyTFLite
        Boost::filesystem
        Eigen3::Eigen
        glog::glog
        TensorFlowLite::TensorFlowLite
        ${OpenCV_LIBS}
        ZLIB::ZLIB)
target_include_directories(EasyTFLite PUBLIC src)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()