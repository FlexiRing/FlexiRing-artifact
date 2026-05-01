//#include "gesture_model_quantized.h" 
#include "gesture_model.h"

#include "am_util.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace {

using GestureOpResolver = tflite::MicroMutableOpResolver<15>;

TfLiteStatus RegisterGestureOps(GestureOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExpandDims());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeBilinear());
  TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
   
  return kTfLiteOk;
}


constexpr int kGestureArenaSize = 40 * 1024; 
alignas(16) static uint8_t gesture_arena[kGestureArenaSize];


static GestureOpResolver gesture_op_resolver;
static const tflite::Model* gesture_model = nullptr;
static tflite::MicroInterpreter* gesture_interpreter = nullptr;
} // namespace


TfLiteStatus InitGestureModel() {
    tflite::InitializeTarget();
    
    gesture_model = tflite::GetModel(gesture_model_tflite);
    if (gesture_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema mismatch!");
        return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(RegisterGestureOps(gesture_op_resolver));

    static tflite::MicroInterpreter static_interpreter(
        gesture_model, gesture_op_resolver, gesture_arena, kGestureArenaSize);
    gesture_interpreter = &static_interpreter;

    TF_LITE_ENSURE_STATUS(gesture_interpreter->AllocateTensors());
    
    return kTfLiteOk;
}

static constexpr uint32_t kSystemClockHz = 48732000;

TfLiteStatus RunGestureInference(float* input_imu_data, float* output_vector_float) {
    TfLiteTensor* input = gesture_interpreter->input(0);
    TfLiteTensor* output = gesture_interpreter->output(0);

    if (input->type == kTfLiteUInt8) {
        float input_scale = input->params.scale;
        int input_zp = input->params.zero_point;
        for (int i = 0; i < (180 * 6); ++i) {
          
            input->data.uint8[i] = (uint8_t)(input_imu_data[i] / input_scale + input_zp);
        }
    } else {
        
        for (int i = 0; i < (180 * 6); ++i) input->data.f[i] = input_imu_data[i];
    }

  
    DWT->CYCCNT = 0;
    uint32_t start_cycles = DWT->CYCCNT;
    TF_LITE_ENSURE_STATUS(gesture_interpreter->Invoke());
    uint32_t end_cycles = DWT->CYCCNT;

    am_util_debug_printf("RunGestureInference cycles: %u\r\n", (unsigned)(end_cycles - start_cycles));

    float elapsed_ms = (float)(end_cycles - start_cycles) / (kSystemClockHz / 1000.0f);
    am_util_debug_printf("Elapsed time: %.3f ms\r\n", elapsed_ms);


    if (output->type == kTfLiteUInt8) {
        uint8_t* output_raw = output->data.uint8; 
        float out_scale = output->params.scale;
        int out_zp = output->params.zero_point;

        am_util_debug_printf("%d %f\r\n",out_zp, out_scale);
        for (int i = 0; i < OUTPUT_LENGTH; ++i) {
         
            output_vector_float[i] = (output_raw[i] - out_zp) * out_scale;
        }
    } else {
  
        for (int i = 0; i < OUTPUT_LENGTH; ++i) output_vector_float[i] = output->data.f[i];
    }

    return kTfLiteOk;
}