cmake_minimum_required(VERSION 3.14)

option(BUILD_TESTS "Build the Tests" ON)
option(BUILD_EXAMPLES "Build the Examples" ON)

project(EasyTFLite)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_HOME_DIRECTORY}/cmake/modules/")

find_package(Boost REQUIRED filesystem random program_options)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorFlowLite)

add_library(EasyTFLite src/TFLite.cpp src/EasyTFLite.cpp src/SSD_EasyTFLite.cpp)
target_link_libraries(EasyTFLite
        Boost::filesystem
        Eigen3::Eigen
        glog::glog
        TensorFlowLite::TensorFlowLite
        ${OpenCV_LIBS})
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