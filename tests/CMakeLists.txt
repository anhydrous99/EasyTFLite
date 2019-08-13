add_executable(TFLite_tests TFLiteTest.cpp)
target_link_libraries(TFLite_tests GTest::GTest Boost::random EasyTFLite)
add_test(NAME TFLite_tests COMMAND TFLite_tests)