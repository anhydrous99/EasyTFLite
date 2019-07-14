//
// Created by Armando Herrera on 2019-07-12.
//

#include "TFLite.h"

#include <tensorflow/lite/kernels/register.h>
#include <boost/filesystem.hpp>

static void model_path_checker(const boost::filesystem::path &model_path) {
    if (!boost::filesystem::exists(model_path))
        LOG(FATAL) << "Error: Couldn't find model - " << model_path << '\n';
    if (model_path.extension() != ".tflite")
        LOG(FATAL) << "Error: model doesn't have .tflite extension\n";
}

TFLite::TFLite(const boost::filesystem::path &model_path) {
    // Check if file exists
    model_path_checker(model_path);

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n";

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n";

    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n";
}

TFLite::TFLite(const boost::filesystem::path &model_path, const tflite::OpResolver &op_resolver) {
    // Check if file exists
    model_path_checker(model_path);

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n";

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::InterpreterBuilder builder(*model, op_resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n";

    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n";
}

TFLite::TFLite(const boost::filesystem::path &model_path, const ExternalContextPair &external_context) {
    // Check if file exists
    model_path_checker(model_path);

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n";

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n";

    // Setting external context
    interpreter->SetExternalContext(external_context.type, external_context.ctx);
    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n";
}

TFLite::TFLite(const boost::filesystem::path &model_path, const ExternalContextPair &external_context,
               const tflite::OpResolver &op_resolver) {
    // Check if file exists
    model_path_checker(model_path);

    // Build model
    LOG(INFO) << "Building model from file\n";
    model = tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str(), &error_reporter);
    if (model == nullptr)
        LOG(FATAL) << "Error: Couldn't Build FlatBufferModel from file\n";

    // Build interpreter
    LOG(INFO) << "Building interpreter from model\n";
    tflite::InterpreterBuilder builder(*model, op_resolver);
    auto res = builder(&interpreter);
    if (interpreter == nullptr || res != kTfLiteOk)
        LOG(FATAL) << "Error: Couldn't Build Interpreter from FlatBufferModel\n";

    // Setting external context
    interpreter->SetExternalContext(external_context.type, external_context.ctx);
    // Allocate tensor buffers.
    LOG(INFO) << "Allocating tensor buffers\n";
    if (interpreter->AllocateTensors() != kTfLiteOk)
        LOG(FATAL) << "Couldn't allocate tensor buffers\n";
}

std::vector<int> TFLite::input_tensors() {
    return interpreter->inputs();
}

std::vector<int> TFLite::output_tensors() {
    return interpreter->outputs();
}

std::vector<int> TFLite::get_tensor_dims(int tensor_index) {
    TfLiteIntArray *dims = interpreter->tensor(tensor_index)->dims;
    std::vector<int> output(dims->size);
    for (int i = 0; i < dims->size; i++)
        output[i] = dims->data[i];
    return output;
}

TfLiteType TFLite::get_tensor_type(int tensor_index) {
    return interpreter->tensor(tensor_index)->type;
}

int TFLite::get_tensor_element_count(int tensor_index) {
    std::vector<int> dims = get_tensor_dims(tensor_index);
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
}
