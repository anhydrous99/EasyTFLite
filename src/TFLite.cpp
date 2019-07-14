//
// Created by Armando Herrera on 2019-07-12.
//

#include "TFLite.h"

#include <tensorflow/lite/kernels/register.h>
#include <boost/filesystem.hpp>
#include <boost/stacktrace.hpp>
#include <glog/logging.h>

typedef boost::stacktrace::stacktrace stacktrace;

TFLite::TFLite(const boost::filesystem::path &model_path) {
    if (!boost::filesystem::exists(model_path))
        LOG(FATAL) << "Error: Couldn't find model - " << model_path << '\n' << stacktrace();
    if (model_path.extension() != ".tflite")
        LOG(FATAL) << "Error: model doesn't have .tflite extension\n" << stacktrace();

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n" << stacktrace();

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n" << stacktrace();

    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n" << stacktrace();
}

TFLite::TFLite(const boost::filesystem::path &model_path, const tflite::OpResolver &op_resolver) {
    if (!boost::filesystem::exists(model_path))
        LOG(FATAL) << "Error: Couldn't find model - " << model_path << '\n' << stacktrace();
    if (model_path.extension() != ".tflite")
        LOG(FATAL) << "Error: model doesn't have .tflite extension\n" << stacktrace();

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n" << stacktrace();

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::InterpreterBuilder builder(*model, op_resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n" << stacktrace();

    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n" << stacktrace();
}

std::vector<int> TFLite::input_tensors() {
    return interpreter->inputs();
}

std::vector<int> TFLite::output_tensors() {
    return interpreter->outputs();
}
