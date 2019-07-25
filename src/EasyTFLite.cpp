//
// Created by Armando Herrera on 2019-07-20.
//

#include "EasyTFLite.h"

std::vector<float *> EasyTFLite::run_inference_ptrs(const cv::Mat &image) {
    // Assuming a scale between -1 and 1
    std::function<float(unsigned char)> scale_func = [](unsigned char x) -> float {
        return static_cast<float>(x) / 127.5 - 1;
    };
    return run_inference_ptrs(image, scale_func);
}
