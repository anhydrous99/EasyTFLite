//
// Created by Armando Herrera on 7/27/19.
//

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <SSD_EasyTFLite.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Parses label text file
std::vector<std::string> label_parse(const fs::path &label_path) {
    std::string line;
    std::vector<std::string> output;
    std::ifstream file(label_path.string());
    if (file.is_open()) {
        while (std::getline(file, line)) {
            output.push_back(line);
        }
    } else
        LOG(FATAL) << "Error: Couldn't open text file\n";
    return output;
}

int main(int argc, char **argv) {
    // Init google logging
    google::InitGoogleLogging(argv[0]);

    // Get Arguments
    float threshold = 0.6;
    std::string videosource("0");
    fs::path project_path(fs::current_path().parent_path());
    fs::path model_path(project_path.string() + "/examples/objectdetection/detect.tflite");
    fs::path label_path(project_path.string() + "/examples/objectdetection/labelmap.txt");
    fs::path output(project_path.string() + "/examples/objectdetection/output.mjpg");

    po::options_description description("An EasyTFLite Single-Shot Object Detection example");
    description.add_options()
            ("model", po::value<fs::path>(&model_path)->default_value(model_path),
             "Path to tensorFlow lite flatbuffer model")
            ("source", po::value<std::string>(&videosource)->default_value(videosource),
             "The video source")
            ("labels", po::value(&label_path)->default_value(label_path),
             "The path to the labels")
            ("threshold", po::value<float>(&threshold)->default_value(threshold),
             "Detection Threshold")
            ("output", po::value(&output)->default_value(output),
             "The path to the output video")
            ("help", "Produce help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, description), vm);
    if (vm.count("help")) {
        std::cout << description << "\n";
        return 0;
    }

    // Show selected arguments
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Video source: " << videosource << std::endl;

    // Get labels
    auto labels = label_parse(label_path);

    // Set colors
    cv::Scalar label_text_color(255, 255, 255);
    cv::Scalar box_color(255, 128, 0);

    // Init model
    SSD_EasyTFLite model(model_path);

    // Init video capture
    cv::VideoCapture cap;
    if (videosource == "0")
        cap.open(0);
    else
        cap.open(videosource);

    // Init video writer
    cv::VideoWriter writer(output.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(cv::CAP_PROP_FPS),
                           cv::Size((int) cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH),
                                    (int) cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT)));

    // Check if video capture was successfully opened
    if (!cap.isOpened())
        LOG(FATAL) << "Error: Video capture couldn't be opened\n";

    cv::Mat frame;
    while (cap.read(frame)) {
        // Run model inference on captured frame
        // First tensor contains the location data
        // Second tensor contains the detected classes
        // Third tensor contains the score for the classes
        // Fourth tensor contains the number of detections
        std::array<Eigen::Tensor<float, 2>, 4> model_output = model.run_inference(frame);

        // Get and loop the number of detections
        int n_detections = static_cast<int>(model_output[3](0, 0));
        for (int i = 0; i < n_detections; i++) {

            // Get the score for the object
            float score = model_output[2](0, i);

            // Filter out scores under threshold
            if (score < threshold)
                continue;

            // Calculate scaled score
            float scaled_score = (score - threshold) * 100 / (threshold - score);

            // Set background score color
            cv::Scalar label_background_color(0, static_cast<int>(scaled_score), 75);

            // Get class & prepare the class label
            int cls = static_cast<int>(model_output[1](0, i));
            std::string label = labels[cls] + " (" + std::to_string(scaled_score) + "%)";

            int box_top = model_output[0](i, 0);
            int box_left = model_output[0](i, 1);
            int box_bottom = model_output[0](i, 2);
            int box_right = model_output[0](i, 3);

            // Draw rectangle of detection
            cv::rectangle(frame, cv::Point(box_left, box_top), cv::Point(box_right, box_bottom), box_color, 2);

            // Draw the classification string just above and to the left of the rectangle
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            baseline += 1;

            int label_left = box_left;
            int label_top = box_top - label_size.height;
            if (label_top < 1)
                label_top = 1;
            int label_right = label_left + label_size.width;
            int label_bottom = label_top + label_size.height;
            cv::rectangle(frame, cv::Point(label_left - 1, label_top - 1), cv::Point(label_right + 1, label_bottom + 1),
                          label_background_color, -1);
            cv::putText(frame, label, cv::Point(label_left, label_bottom), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        label_text_color, 1);
        }

        // display text to let user know how to exit
        cv::rectangle(
                frame,
                cv::Rect(0, 0, 100, 15),
                cv::Scalar(255, 0, 0)
        );
        cv::putText(
                frame,
                "q to Quit",
                cv::Point(10, 12),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                cv::Scalar(255, 0, 0)
        );

        writer.write(frame);
    }

    // Clean up
    cap.release();
    writer.release();
    return 0;
}