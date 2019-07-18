//
// Created by aherrera on 7/17/19.
//

#include "TFLite.h"
#include "gtest/gtest.h"

#include <string>
#include <memory>

namespace {
    class TFLiteTest {
    protected:
        // Per-test-suite set-up
        static void SetUpTestSuite() {
            tflite = new TFLite(boost::filesystem::path(""));
            tflite_quant = new TFLite(boost::filesystem::path(""));
        }

        // Per-test-suite tear-down.
        static void TearDownTestSuite() {
            delete tflite;
            delete tflite_quant;
            tflite = nullptr;
            tflite_quant = nullptr;
        }

        static TFLite *tflite;
        static TFLite *tflite_quant;
    };

    TFLite *TFLiteTest::tflite = nullptr;
    TFLite *TFLiteTest::tflite_quant = nullptr;

    TEST_F(TFLiteTest, InputListTest) {
        //
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}