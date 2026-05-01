#include <cstdint>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern unsigned int gesture_model_quantized_tflite_len;
extern const unsigned char gesture_model_quantized_tflite[];

//TfLiteStatus RunGestureInference(float* input_imu_data, uint8_t* output_vector);
//TfLiteStatus InitGestureModel();