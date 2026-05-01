#include <cstdint>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define OUTPUT_LENGTH 256

extern unsigned int gesture_model_tflite_len;
extern const unsigned char gesture_model_tflite[];

TfLiteStatus RunGestureInference(float* input_imu_data, float* output_vector);
TfLiteStatus InitGestureModel();