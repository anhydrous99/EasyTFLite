cmake_minimum_required(VERSION 3.14)

option(BUILD_TESTS "Build the Tests" ON)
option(BUILD_EXAMPLES "Build the Examples" ON)

include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://vmobipypi.blob.core.windows.net/pypi-blob/hunter.tar.gz"
        SHA1 "4acb38f0ed2b57d9c85727152e10b3b014668e95"
)

project(EasyTFLite)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}/cmake/modules/")

hunter_add_package(Boost COMPONENTS filesystem random program_options)
hunter_add_package(Eigen)
hunter_add_package(glog)
hunter_add_package(GTest)
hunter_add_package(OpenCV)
hunter_add_package(ZLIB)

find_package(Boost CONFIG REQUIRED filesystem random program_options)
find_package(Eigen3 CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorFlowLite)
find_package(ZLIB CONFIG REQUIRED)

add_library(EasyTFLite src/TFLite.cpp src/EasyTFLite.cpp src/SSD_EasyTFLite.cpp)
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

if (BUILD_EXAMPLES)
    add_executable(GalaxyClassification examples/galaxyclassification/GalaxyClassification.cpp)
    target_link_libraries(GalaxyClassification EasyTFLite Boost::program_options)

    add_executable(SSD_ObjectDetection examples/objectdetection/SSD_ObjectDetection.cpp)
    target_link_libraries(SSD_ObjectDetection EasyTFLite Boost::program_options)
endif ()