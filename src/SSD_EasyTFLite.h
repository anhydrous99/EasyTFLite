//
// Created by armandoh on 7/24/19.
//

#ifndef EASYTFLITE_SSD_EASYTFLITE_H
#define EASYTFLITE_SSD_EASYTFLITE_H

#include "EasyTFLite.h"

//! The SSD_EasyTFLite class inherits EasyTFLite whose objective is to have single function inference for SSD Object Detection
/*!
 * SSD_EasyTFLite, with this class you can run inferencing using the Single Shot MultiBox Detector Tensorflow Lite models.
 * To use this, it is recommended to use a Tensorflow provided model and retain for your purposes. You can, however,
 * mimic the input and output tensors of those model, and that would work. This class assumes input range of -1 and 1.
 */
class SSD_EasyTFLite : private EasyTFLite {
    //! Whether the model is quantized or not
    bool quant_model;

public:
    /*!
    * Initializes SSD_EasyTFLite
    * @param model_path The path to a Single Shot MultiBox Detector Tensorflow Lite Flatbuffer Model
    */
    explicit SSD_EasyTFLite(const boost::filesystem::path &model_path);

    /*!
     * Runs inferencing, output results in four Rank 4 tensors, the first float of the first tensor contains the
     * locations of the detected objects in [10][4], the second contains the classes, the third contains the scores for
     * the classes, and, finally, the last and fourth tensors contains the number of detection in the first float and
     * only float of the tensor.
     * @param input_image OpenCV's Mat image to run inference on
     * @return A array of 4 eigen tensors
     */
    std::array<Eigen::Tensor<float, 2>, 4> run_inference(const cv::Mat &input_image);
};


#endif //EASYTFLITE_SSD_EASYTFLITE_H
