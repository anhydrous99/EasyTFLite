# Introduction
The goal of this project is to make Tensorflow Lite easier to use. To achieve this, this project will wrap the C++
Tensorflow Lite API simplifying it's use. Additionally, simplifying integration with libraries such as OpenCV and Eigen.
A goal for this project will be for it to be as cross-platform as possible.

# Compiling
The project uses CMake with the Hunter cross-platform package manager, so when using cmake most of the C++
dependencies are downloaded and compiled on-the-fly. Tensorflow Lite on the other hand needs to be compiled separately.
This is because Tensorflow Lite needs to be compiled using bazel.

The version of Tensorflow I currently use is the r2.0 branch on 
[Tensorflow's github page](https://github.com/tensorflow/tensorflow).
The version of bazel I use to compile it is version 0.25.2 and you can download it from 
the [bazel github release page](https://github.com/bazelbuild/bazel/releases).

## Ubuntu/Debian
On Ubuntu/Debian you can run the following commands. Now, in a arbitrary directory and assuming you have bazel 
installed you can run the following commands to build Tensorflow Lite.
```
sudo apt install cmake git build-essential
git clone -b r2.0 https://github.com/tensorflow/tensorflow
cd tensorflow
bazel build //tensorflow/lite:libtensorflowlite.so
```
Now you need to keep in mind where the tensorflow directory is and run the following commands, in another arbitrary 
directory:
```
git clone https://github.com/anhydrous99/EasyTFLite
cd EasyTFLite
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path_to_tensorflow>
make
```

## Mac OS X Mojave
I assume that you XCode, cmake, brew. Just a quick disclaimer on my mac I have SIP disabled and I am not to sure 
if having it enables affects the install process. Anyway, to build tensorflow lite:
```
git clone -b r2.0 https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
```
Follow the prompt then run:
```
bazel build //tensorflow/lite:libtensorflowlite.so
```
Then you can run the same commands as you would in Ubuntu, in another arbitrary directory:
```
git clone https://github.com/anhydrous99/EasyTFLite
cd EasyTFLite
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path_to_tensorflow>
make
```
