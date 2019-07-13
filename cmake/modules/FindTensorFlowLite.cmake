#
# FindTensorFlowLite.cmake
# This module finds libtensorflowlite.so and it's include directories
# TensorFlowLite::TensorFlowLite
#

if (NOT TENSORFLOW_PATH)
    message(FATAL_ERROR "TENSORFLOW_PATH must point to your TensorFlow source build")
endif ()

set(TENSORFLOWLITE_INCLUDE_DIR
        "${TENSORFLOW_PATH}"
        "${TENSORFLOW_PATH}/tensorflow"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/eigen"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/absl"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/gemmlowp"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/neon_2_sse"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/farmhash/src"
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include")

find_library(TENSORFLOWLITE_LIBRARY
        NAMES tensorflow-lite libtensorflow-lite
        HINTS
        "${TENSORFLOW_PATH}/tensorflow/lite/tools/make/gen/osx_x86_64/lib"
)

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
