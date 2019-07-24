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
The version of bazel I use to compile it is version 0.26.0 and you can download it from 
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

### Installing bazel
In ubuntu/debian the package manager installs any dependencies bazel needs, so you can run the following:
```
wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel_0.26.0-linux-x86_64.deb
sudo apt install ./bazel_0.26.0-linux-x86_64.deb
```

## Arch Linux/Manjaro
On arch/manjaro you can run the following to get bazel installed, build tensorflow lite, and build EasyTFLite:
```
sudo pacman -S base-devel python3 python3-venv git cmake jdk-openjdk
wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel-0.26.0-installer-linux-x86_64.sh
sudo sh bazel-0.26.0-installer-linux-x86_64.sh
rm bazel-0.26.0-installer-linux-x86_64.sh
git clone -b r2.0 https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
bazel build //tensorflow/lite:libtensorflowlite.so
cd ..
git clone https://github.com/anhydrous99/EasyTFLite
cd EasyTFLite
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path_to_tensorflow>
make
```

## Mac OS X Mojave
Unfortunately, the dynamic build of tensorflow lite, using bazel is broken, on Mac OS X Mojave. So, you can use the 
static build. Unfortunately, again, the static tensorflow lite build on the latest versions of tensorflow is broken. 
But, however, you can use tensorflow version 1.13.\*. Be sure to have cmake, xcode, git, and all that jazz installed. 
The process is:
```
git clone -b r1.13 https://github.com/tensorflow/tensorflow
cd tensorflow
sh ./tensorflow/lite/tools/make/download_dependencies.sh
make -f tensorflow/lite/tools/make/Makefile
```
Now to build EasyTFLite with the extra cmake flag `STATIC_TENSORFLOWLITE` set to `ON`
```
git clone https://github.com/anhydrous99/EasyTFLite
cd EasyTFLite
mkdir build && cd build
cmake .. -DTENSORFLOW_PATH=<path_to_tensorflow> -DSTATIC_TENSORFLOWLITE=ON
make
```