//
// Created by Armando Herrera on 2019-07-20.
//

#ifndef EASYTFLITE_EASYTFLITE_H
#define EASYTFLITE_EASYTFLITE_H

#include "TFLite.h"

#include <functional>
#include <opencv2/core/mat.hpp>

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
     * @param image OpenCV's Mat image to run inference on
     * @param scale_func Your custom scale and preprocesses function, the input of the function must be an unsigned char
     * and the output a float. This function is applied to all elements of image after it has been scaled to the size
     * of model.
     * @return A vector of pointers to the output tensors
     */
    std::vector<float *>
    run_inference_ptrs(const cv::Mat &image, const std::function<float(unsigned char)> &scale_func);

    /*!
     * Runs inference on an OpenCV Mat image, returns a vector of pointers to the output data. You can use this function
     * to define a custom preprocess function between the image being scaled and the image data being sent to the
     * interpreter for inference. The model must only have a single input and the input must be a rank 4 Tensor,
     * [1, width, height, channels].
     * @param image OpenCV's Mat image to run inference on
     * @param preprocess_func Your custom preprocess function, the input of the function must be an cv::Mat
     * and the output a vector of floats containing the elements converted to float flattened to a single dimension, C
     * style.
     * @return A vector of pointers to the output tensors
     */
    std::vector<float *>
    run_inference_ptrs(const cv::Mat &image, const std::function<std::vector<float>(cv::Mat)> &preprocess_func);

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
};


#endif //EASYTFLITE_EASYTFLITE_H
