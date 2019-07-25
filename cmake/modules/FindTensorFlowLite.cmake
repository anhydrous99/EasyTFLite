#
# FindTensorFlowLite.cmake
# This module finds libtensorflowlite.so and it's include directories
# TensorFlowLite::TensorFlowLite
#

if (NOT TENSORFLOW_PATH)
    message(FATAL_ERROR "TENSORFLOW_PATH must point to your TensorFlow source build")
endif ()

if (NOT STATIC_TENSORFLOWLITE)
    set(BAZEL_EXTERNAL_PATH "${TENSORFLOW_PATH}/bazel-tensorflow/external")

    set(TENSORFLOWLITE_INCLUDE_DIR
            "${TENSORFLOW_PATH}"
            "${TENSORFLOW_PATH}/tensorflow"
            "${BAZEL_EXTERNAL_PATH}/eigen_archive"
            "${BAZEL_EXTERNAL_PATH}/gemmlowp"
            "${BAZEL_EXTERNAL_PATH}/arm_neon_2_x86_sse"
            "${BAZEL_EXTERNAL_PATH}/farmhash_archive/src"
            "${BAZEL_EXTERNAL_PATH}/flatbuffers/include")

    find_library(TENSORFLOWLITE_LIBRARY
            NAMES tensorflowlite libtensorflowlite
            HINTS
            "${TENSORFLOW_PATH}/bazel-bin/tensorflow/lite"
    )
else ()
    set(TENSORFLOWLITE_MAKE_PATH "${TENSORFLOW_PATH}/tensorflow/lite/tools/make")

    set(TENSORFLOWLITE_INCLUDE_DIR
            "${TENSORFLOW_PATH}"
            "${TENSORFLOW_PATH}/tensorflow"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads/eigen"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads/gemmlowp"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads/neon_2_sse"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads/farmhash/src"
            "${TENSORFLOWLITE_MAKE_PATH}/downloads/flatbuffers/include")

    set(TENSORFLOWLITE_HINTS_PATH "${TENSORFLOWLITE_MAKE_PATH}/gen/linux_x86_64/lib")
    if (APPLE)
        set(TENSORFLOWLITE_HINTS_PATH "${TENSORFLOWLITE_MAKE_PATH}/gen/osx_x86_64/lib")
    endif ()

    find_library(TENSORFLOWLITE_LIBRARY
            NAMES tensorflow-lite libtensorflow-lite
            HINTS
            ${TENSORFLOWLITE_HINTS_PATH})
endif ()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TENSORFLOWLITE
        REQUIRED_VARS TENSORFLOWLITE_LIBRARY TENSORFLOWLITE_INCLUDE_DIR)

if (TENSORFLOWLITE_FOUND)
    set(TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARY})
    set(TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIR})
    if(NOT TARGET TensorFlowLite::TensorFlowLite)
        add_library(TensorFlowLite::TensorFlowLite UNKNOWN IMPORTED)
        set_target_properties(TensorFlowLite::TensorFlowLite PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOWLITE_INCLUDE_DIRS}")
        set_property(TARGET TensorFlowLite::TensorFlowLite APPEND PROPERTY IMPORTED_LOCATION "${TENSORFLOWLITE_LIBRARY}")
    endif()
endif ()

mark_as_advanced(TENSORFLOWLITE_INCLUDE_DIR TENSORFLOWLITE_LIBRARY)
