# Introduction
The goal of this project is to make Tensorflow Lite easier to use. To achieve this, this project will wrap the C++
Tensorflow Lite API simplifying it's use. Additionally, simplifying integration with libraries such as OpenCV and Eigen.
A goal for this project will be for it to be as cross-platform as possible.

# Compiling
The project uses the CMake in it's build process. A couple of requirements are needed to build the project:
* Tensorflow Lite
* OpenCV
* Google Logging
* Eigen V3
* Boost - filesystem

The examples and testing requires:
* Google Testing
* Boost - random, program_options

## Manjaro/Arch Linux
```
sudo pacman -S base-devel bazel opencv google-glog eigen vtk hdf5 boost gtest graphviz doxygen jdk-openjdk
export PATH=/usr/lib/jvm/java-12-openjdk/bin${PATH:+:${PATH}}
git clone -b r2.0 https://github.com/tensorflow/tensorflow.git
git clone https://github.com/anhydrous99/EasyTFLite
cd tensorflow
bazel build //tensorflow/lite:libtensorflowlite.so
cd ../EasyTFLite
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path to tensorflow>
make
```
### Build docs
From the project directory
```
cd docs
doxygen
```
### Testing
From the project directory
```
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path to tensorflow>
make
make test
```