//
// Created by Armando Herrera on 2019-07-12.
//

#ifndef EASYTFLITE_TFLITE_H
#define EASYTFLITE_TFLITE_H

#include <map>
#include <memory>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/multi_array.hpp>
#include <boost/variant/variant.hpp>
#include <boost/mpl/contains.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/op_resolver.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

//! A struct that contains the TfLite context type and a pointer to the TfLite Context
struct ExternalContextPair {
    TfLiteExternalContextType type;
    TfLiteExternalContext *ctx;
};

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
     * @param model_path A boost path object containing the path to the Tensorflow Lite Flatbuffer model
     */
    explicit TFLite(const boost::filesystem::path &model_path);

    /*!
     * You can use this function, and define your own OpResolver for custom operators.
     * @param model_path A boost path object containing the path to the Tensorflow Lite Flatbuffer model
     * @param op_resolver An instance that implements the OpResolver interface. (You can have a custom
     * Resolver with custom ops)
     */
    TFLite(const boost::filesystem::path &model_path, const tflite::OpResolver &op_resolver);

    /*!
     * You can use this constructor to also set the external context (e.g. EdgeTPU)
     * @param model_path A boost path object containing the path to the Tensorflow Lite Flatbuffer model
     * @param external_context The external context (e.g. EdgeTPU)
     */
    TFLite(const boost::filesystem::path &model_path, const ExternalContextPair &external_context);

    /*!
     * You can use this constructor to also set the external context and a custom OpResolver
     * @param model_path A boost path object containing the path to the Tensorflow Lite Flatbuffer model
     * @param external_context The external context (e.g. EdgeTPU)
     * @param op_resolver An instance that implements the OpResolver interface. (You can have a custom
     * Resolver with custom ops)
     */
    TFLite(const boost::filesystem::path &model_path, const ExternalContextPair &external_context,
           const tflite::OpResolver &op_resolver);

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
     * Get dimension of tensor
     * @param tensor_index Index of tensor to get dimensions for
     * @return A vector containing dimension of tensor
     */
    std::vector<int> get_tensor_dims(int tensor_index);

    /*!
     * Get the tensor type
     * @param tensor_index Index of tensor to get type for
     * @return A TfLiteType enum denoting type (Probably either it is kTfLiteFloat32 denoting 32 bit float or
     * kTfLiteUint8 denoting unsigned 8 bit integer)
     */
    TfLiteType get_tensor_type(int tensor_index);

    /*!
     * Gets the number of elements in a tensor
     * @param tensor_index Index of tensor
     * @return The number of elements in the tensor with tensor_index
     */
    int get_tensor_element_count(int tensor_index);

    /*!
     * Fills input tensor
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @param data Pointer to data
     * @param tensor_index Index of the input tensor to fill
     * @param n_elements Number of elements in data
     */
    template<typename T>
    void fill_input_tensor(T *data, int tensor_index, int n_elements) {
        // Stops if T is not uint8_t or float
        BOOST_STATIC_ASSERT(boost::mpl::contains<boost::variant<uint8_t, float>::types, T>::value);
        auto tensor_ptr = interpreter->typed_tensor<T>(tensor_index);
        std::copy(data, data + n_elements, tensor_ptr);
    }

    /*!
     * Fill input tensor from an Eigen::Tensor
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @tparam Rank Tensor rank
     * @param tensor The tensor in the form of a Eigen::Tensor
     * @param tensor_index Index of the input tensor to fill
     */
    template<typename T, int Rank>
    void fill_input_tensor(const Eigen::Tensor<T, Rank> &tensor, int tensor_index) {
        int n_elements = static_cast<int>(tensor.size());
        T *input_tensor_ptr = tensor.data();
        fill_input_tensor(input_tensor_ptr, tensor_index, n_elements);
    }

    /*!
     * Fill input tensor from a boost::multi_array
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @tparam Rank Tensor rank
     * @param tensor The tensor in the form of a boost::multi_array
     * @param tensor_index Index of the input tensor to fill
     */
    template<typename T, unsigned long Rank>
    void fill_input_tensor(const boost::multi_array<T, Rank> &tensor, int tensor_index) {
        int n_elements = tensor.num_elements();
        T *input_tensor_ptr = tensor.data();
        fill_input_tensor(input_tensor_ptr, tensor_index, n_elements);
    }

    /*!
     * Fill all input tensors from a hashtable of tensors
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @tparam Rank Tensor rank
     * @param tensors A map where the key is the tensor index and the value is the Eigen::Tensor tensor
     */
    template<typename T, int Rank>
    void fill_input_tensors(const std::map<int, Eigen::Tensor<T, Rank>> &tensors) {
        for (const auto &tensor : tensors) {
            fill_input_tensor(tensor.second, tensor.first);
        }
    }

    /*!
     * Fill all input tensors from a hashtable of tensors
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not.
     * @tparam Rank Tensor rank
     * @param tensors A map where the key is the tensor index and the value is the boost::multi_array tensor
     */
    template<typename T, unsigned long Rank>
    void fill_input_tensors(const std::map<int, boost::multi_array<T, Rank>> &tensors) {
        for (const auto &tensor : tensors) {
            fill_input_tensor(tensor.second, tensor.first);
        }
    }
};


#endif //EASYTFLITE_TFLITE_H
