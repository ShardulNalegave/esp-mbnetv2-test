
#include <freertos/FreeRTOS.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <bsp/esp-bsp.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_profiler.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "camera.h"
#include "model.h"

static const char* TAG = "human_detect_app";

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    tflite::MicroProfiler profiler;

    constexpr int kTensorArenaSize = 1.5 * 1024 * 1024;
    static uint8_t* tensor_arena;

    TfLiteTensor* input;
    TfLiteTensor* output;
}

int8_t quantize(float val) {
    auto zero_point = input->params.zero_point;
    auto scale = input->params.scale;
    auto quant = (val / scale) + zero_point;
    if (quant < -128) quant = -128;
    if (quant > 127) quant = 127;
    return quant;
}

float dequantize(int8_t val) {
    auto zero_point = output->params.zero_point;
    auto scale = output->params.scale;
    return (val - zero_point) * scale;
}

extern "C" void app_main(void)
{
    if(app_camera_init() != 0) {
        ESP_LOGE(TAG, "Couldn't initialize camera");
        return;
    }
    ESP_LOGI(TAG, "Initialized Camera");

    model = tflite::GetModel(human_detect_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    }

    if (tensor_arena == NULL) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }

    if (tensor_arena == NULL) {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }

    ESP_LOGI(TAG, "Allocated memory for Tensor Arena");

    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddPad();

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr, &profiler
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    ESP_LOGI(TAG, "Allocated Tensors");

    input = interpreter->input(0);
    output = interpreter->output(0);

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            return;
        }

        // lv_draw_sw_rgb565_swap(cropped_img, 224 * 224);

        int x_offset = (320 - 224) / 2;
        int y_offset = (240 - 224) / 2;

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int in_x = x + x_offset;
                int in_y = y + y_offset;
                uint16_t pixel = ((uint16_t*)fb)[in_y * 224 + in_x];

                float r = ((pixel >> 11) & 0x1F) * (255.0f / 31.0f);
                float g = ((pixel >> 5)  & 0x3F) * (255.0f / 63.0f);
                float b = (pixel & 0x1F) * (255.0f / 31.0f);

                float normalized_r = (r / 127.5f) - 1.0f;
                float normalized_g = (g / 127.5f) - 1.0f;
                float normalized_b = (b / 127.5f) - 1.0f;

                int input_offset = (x * 224 + y) * 3;
                input->data.int8[input_offset] = quantize(normalized_r);
                input->data.int8[input_offset + 1] = quantize(normalized_g);
                input->data.int8[input_offset + 2] = quantize(normalized_b);
            }
        }

        esp_camera_fb_return(fb);

        if (interpreter->Invoke() != kTfLiteOk) {
            MicroPrintf("Invoke() failed");
        }
        
        float score = dequantize(output->data.int8[0]);
        printf("Score = %f\n", score);
        // if (score >= 0.5) {
        //     printf("Human\n");
        // } else {
        //     printf("No-Human\n");
        // }
    }
}
