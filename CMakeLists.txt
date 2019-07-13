cmake_minimum_required(VERSION 3.14)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.204.tar.gz"
        SHA1 "32cfed254da901f6f184027d530d8da47e035b85"
)

project(EasyTFLite)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}/cmake/modules/")

hunter_add_package(Boost COMPONENTS filesystem)
hunter_add_package(glog)
hunter_add_package(GTest)
hunter_add_package(OpenCV)
hunter_add_package(ZLIB)

find_package(Boost CONFIG REQUIRED filesystem)
find_package(glog CONFIG REQUIRED)
find_package(GMock CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorFlowLite)
find_package(ZLIB CONFIG REQUIRED)

add_library(EasyTFLite src/TFLite.cpp src/TFLite.h)
target_link_libraries(EasyTFLite
        Boost::filesystem
        glog::glog
        GMock::main
        TensorFlowLite::TensorFlowLite
        ${OpenCV_LIBS}
        ZLIB::ZLIB)
