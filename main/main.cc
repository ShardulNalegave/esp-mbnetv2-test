
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
    return (val / scale) + zero_point;
}

float dequantize(int8_t val) {
    auto zero_point = output->params.zero_point;
    auto scale = output->params.scale;
    return (val - zero_point) * scale;
}

void crop_img(const uint16_t *input, int in_w, int in_h, uint16_t *output, int out_w, int out_h) {
    int x_offset = (in_w - out_w) / 2;
    int y_offset = (in_h - out_h) / 2;

    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            output[i * out_w + j] = input[(i + y_offset) * in_w + (j + x_offset)];
        }
    }
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

    uint16_t* cropped_img = (uint16_t*)heap_caps_malloc(224 * 224 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!cropped_img) {
        printf("Failed to allocate memory for cropped image\n");
        return;
    }

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            return;
        }

        crop_img((const uint16_t *)fb->buf, 320, 240, cropped_img, 224, 224);
        lv_draw_sw_rgb565_swap(cropped_img, 224 * 224);
        esp_camera_fb_return(fb);

        for (int i = 0; i < 224 * 224; i++) {
            uint16_t pixel = ((uint16_t *) (cropped_img))[i];
            uint8_t hb = pixel & 0xFF;
            uint8_t lb = pixel >> 8;

            uint8_t r = (lb & 0x1F) << 3;
            uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
            uint8_t b = (hb & 0xF8);

            uint16_t inp_offset = i * 3;
            input->data.int8[inp_offset] = quantize(r);
            input->data.int8[inp_offset + 1] = quantize(g);
            input->data.int8[inp_offset + 2] = quantize(b);
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            MicroPrintf("Invoke() failed");
        }
        
        float score = dequantize(output->data.int8[0]);
        if (score >= 0.5) {
            printf("Human\n");
        } else {
            printf("No-Human\n");
        }
    }
}
