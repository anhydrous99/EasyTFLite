//
// Created by Armando Herrera on 2019-07-12.
//

#ifndef EASYTFLITE_TFLITE_H
#define EASYTFLITE_TFLITE_H

#include <memory>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/multi_array.hpp>
#include <boost/variant/variant.hpp>
#include <boost/mpl/contains.hpp>
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

    /*!
     * Gets indexes of all input tensors
     * @return A vector of ints indicating input tensor indexes
     */
    std::vector<int> input_tensors();

    /*!
     * Gets indexes of all output tensors
     * @return A vector of ints indicating output tensor indexes
     */
    std::vector<int> output_tensors();

    /*!
     * Fills input tensor
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @param data Pointer to data
     * @param tensor_index Index of input tensor
     * @param n_elements Number of elements in data
     */
    template<typename T>
    void fill_input_tensor(T* data, int tensor_index, int n_elements) {
        // Stops if T is not uint8_t or float
        BOOST_STATIC_ASSERT(boost::mpl::contains<boost::variant<uint8_t, float>::types, T>::value);
        auto tensor_ptr = interpreter->typed_input_tensor<T>(tensor_index);
        for (int i = 0; i < n_elements; i++)
            tensor_ptr[i] = data[i];
    }
};


#endif //EASYTFLITE_TFLITE_H
