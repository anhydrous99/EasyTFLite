//
// Created by Armando Herrera on 2019-07-12.
//

#ifndef EASYTFLITE_TFLITE_H
#define EASYTFLITE_TFLITE_H

#include <memory>
#include <boost/filesystem/path.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/op_resolver.h>

//! The TFLite class wraps Tensorflow Lite
/*!
 * This class abstracts and interfaces with Tensorflow Lite, taking care of any small details required to use it.
 */
class TFLite {
    tflite::StderrReporter error_reporter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

public:
    /*!
     * This function initialized TFLite with the built-in Ops.
     * @param model_path A boost path object containing path to the Tensorflow Lite path.
     */
    explicit TFLite(const boost::filesystem::path &model_path);

    /*!
     * You can use this function, and define your own OpResolver for custom operators.
     * @param model_path A boost path object containing path to the Tensorflow Lite path.
     * @param op_resolver An instance that implements the OpResolver interface. (You can have a custom
     * Resolver with custom ops)
     */
    TFLite(const boost::filesystem::path &model_path, const tflite::OpResolver &op_resolver);
};


#endif //EASYTFLITE_TFLITE_H
