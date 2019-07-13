//
// Created by Armando Herrera on 2019-07-12.
//

#ifndef EASYTFLITE_TFLITE_H
#define EASYTFLITE_TFLITE_H

#include <boost/filesystem/path.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <tensorflow/lite/interpreter.h>

class TFLite {
    boost::scoped_ptr<tflite::Interpreter> model;
};


#endif //EASYTFLITE_TFLITE_H
