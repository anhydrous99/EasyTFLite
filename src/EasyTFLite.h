//
// Created by Armando Herrera on 2019-07-20.
//

#ifndef EASYTFLITE_EASYTFLITE_H
#define EASYTFLITE_EASYTFLITE_H

#include "TFLite.h"

#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

//! The EasyTFLite class inherits TFLite whose objective is convenience and speed of creating a Tensorflow Lite pipeline
/*!
 * This class inherits TFLite and it's constructors while keeping TFLite's functions in a protected state. The objective
 * of this class is to simplify the usage of other libraries with Tensorflow Lite. So, for instance, have
 * A single function that runs inferencing on OpenCV's Mat class.
 */
class EasyTFLite : protected TFLite {
public:
    using TFLite::TFLite;

    /*!
     * Runs inference on an OpenCV Mat image, returns a vector of pointers to output data.
     * The model must only have a single input and that input must be a rank 4 Tensor,
     * [1, width, height, channels]. The image's dimension is scaled to the input of the model and the values are
     * scaled to between -1 and 1.
     * @param image OpenCV's Mat image to run inference on
     * @return A vector of pointers to the output tensors
     */
    std::vector<float *> run_inference_ptrs(const cv::Mat &image);

    /*!
     * Runs inference on an OpenCV Mat image, returns a vector of pointers to the output data. You can use this function
     * to define a custom scale function and preprocess an image between it being scaled and the image data being sent
     * to the interpreter for inference. The model must only have a single input and the input must be a rank 4 Tensor,
     * [1, width, height, channels].
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not
     * @param image OpenCV's Mat image to run inference on
     * @param scale_func Your custom scale and preprocesses function, the input of the function must be an unsigned char
     * and the output a float. This function is applied to all elements of image after it has been scaled to the size
     * of model.
     * @return A vector of pointers to the output tensors
     */
    template<typename T>
    std::vector<T *> run_inference_ptrs(const cv::Mat &image, const std::function<T(unsigned char)> &scale_func) {
        std::vector<int> it = input_tensors();
        std::vector<int> ot = output_tensors();
        int input_index = it[0];

        // Assuming a one input model
        if (it.size() != 1)
            LOG(FATAL) << "Error: OpenCV's Mat inferencing can only be done on models with one input.\n";
        // Assuming a Rank 4 input tensor for the model with the following [1, x, y, c]
        std::vector<int> dims = get_tensor_dims(input_index);
        if (dims.size() != 4)
            LOG(FATAL) << "Error: OpenCV's Mat inferencing requires a model with a Rank 4 input tensor.\n";

        // Resize image
        cv::Mat resized_image;
        cv::Size target_size(dims[1], dims[2]);
        cv::resize(image, resized_image, target_size);

        // Get the number of elements in image
        int total_elements = resized_image.channels() * resized_image.rows * resized_image.cols;
        const unsigned char *resized_image_ptr = resized_image.data;

        // Where converted data will be stored
        std::vector<T> float_data;

        // Run Scale and convert function
        std::transform(resized_image_ptr, resized_image_ptr + total_elements, std::back_inserter(float_data),
                       scale_func);

        // Fill input tensor
        fill_tensor<T>(float_data.data(), input_index);

        // Invoke model
        invoke();

        return get_output_tensor_ptrs<T>();
    }

    /*!
     * Runs inference on an OpenCV Mat image, returns a vector of pointers to the output data. You can use this function
     * to define a custom preprocess function between the image being scaled and the image data being sent to the
     * interpreter for inference. The model must only have a single input and the input must be a rank 4 Tensor,
     * [1, width, height, channels].
     * @tparam T The tensor type, must be uint8_t or float, depending if model is quantized or not
     * @param image OpenCV's Mat image to run inference on
     * @param preprocess_func Your custom preprocess function, the input of the function must be an cv::Mat
     * and the output a vector of floats containing the elements converted to float flattened to a single dimension, C
     * style.
     * @return A vector of pointers to the output tensors
     */
    template<typename T>
    std::vector<T *> run_inference_ptrs(const cv::Mat &image, const std::function<std::vector<T>(cv::Mat)> &preprocess_func) {
        std::vector<int> it = input_tensors();
        std::vector<int> ot = output_tensors();
        int input_index = it[0];

        // Assuming a one input model
        if (it.size() != 1)
            LOG(FATAL) << "Error: OpenCV's Mat inferencing can only be done on models with one input.\n";
        // Assuming a Rank 4 input tensor for the model with the following [1, x, y, c]
        std::vector<int> dims = get_tensor_dims(input_index);
        if (dims.size() != 4)
            LOG(FATAL) << "Error: OpenCV's Mat inferencing requires a model with a Rank 4 input tensor.\n";

        // Resize image
        cv::Mat resized_image;
        cv::Size target_size(dims[1], dims[2]);
        cv::resize(image, resized_image, target_size);

        // Apply preprocess function
        std::vector<T> float_data = preprocess_func(resized_image);

        // Fill input tensor
        fill_tensor<T>(float_data.data(), input_index);

        // Invoke model
        invoke();

        return get_output_tensor_ptrs<T>();
    }

    /*!
     * Runs inference where the input is a vector of pointers that point to the flattened input data and the output is
     * a vector of pointers that point to the output data. It is assumed that the input size or the tensors is correct.
     * @tparam InputType The input tensor data type, it must be uint8_t or float
     * @tparam OutputType The output tensor data type, it must be uint8_t or float
     * @param input_ptr A vector of pointers of type InputType that point to the input data
     * @return A vector of pointers of type OutputType that point to the output data
     */
    template<typename InputType, typename OutputType>
    std::vector<OutputType *> run_inference_ptrs(const std::vector<InputType *> input_ptrs) {
        // Fill input tensors
        fill_input_tensors<InputType>(input_ptrs);
        // Invoke Network
        invoke();
        // Get output tensors
        return get_output_tensor_ptrs<OutputType>();
    }

    /*!
     * Runs inference where the input is a vector of Eigen Tensors that contain the input data and the output is a
     * vector of Eigen Tensors that contain the output data. All input tensors in the model must have the same rank
     * as well as the output tensors must have the same rank. It is in the map to have the template ranks
     * be the largest rank in either the input or output tensors.
     * @tparam InputType The input tensor data type, it must be uint8_t or float
     * @tparam OutputType The output tensor data type, it must be uint8_t or float
     * @tparam InputRank The input tensor Rank, all inputs of model must have the same rank
     * @tparam OutputRank The output tensor Rank, all outputs of model must have the same rank
     * @param input_tensors The input eigen tensors contained in a vector
     * @return The output eigen tensors contained in a vector
     */
    template<typename InputType, typename OutputType, int InputRank, int OutputRank>
    std::vector<Eigen::Tensor<OutputType, OutputRank>> run_inference(const std::vector<Eigen::Tensor<InputType, InputRank>> input_tensors) {
        // Fill input tensors
        fill_input_tensors<InputType, InputRank>(input_tensors);
        // Invoke Network
        invoke();
        // Get output tensors
        return get_output_tensors<OutputType, OutputRank>();
    }
};


#endif //EASYTFLITE_EASYTFLITE_H
