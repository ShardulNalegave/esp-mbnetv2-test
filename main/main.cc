
#include <freertos/FreeRTOS.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <bsp/esp-bsp.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "camera.h"
#include "model.h"

static const char* TAG = "human_detect_app";

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;

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

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *person_indicator = NULL;
static lv_obj_t *label = NULL;

static void create_gui(void) {
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  person_indicator = lv_led_create(lv_scr_act());
  assert(person_indicator);
  lv_obj_align(person_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  lv_led_set_color(person_indicator, lv_palette_main(LV_PALETTE_GREEN));

  label = lv_label_create(lv_scr_act());
  assert(label);
  lv_label_set_text_static(label, "Person detected");
  lv_obj_align_to(label, person_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
  bsp_display_unlock();
}

extern "C" void app_main(void)
{
    if(app_camera_init() != 0) {
        ESP_LOGE(TAG, "Couldn't initialize camera");
        return;
    }
    ESP_LOGI(TAG, "Initialized Camera");

    create_gui();

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
        model, micro_op_resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    ESP_LOGI(TAG, "Allocated Tensors");

    input = interpreter->input(0);
    output = interpreter->output(0);

    uint8_t* rgb888_buf = (uint8_t*)heap_caps_malloc(240 * 240 * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb888_buf) {
        ESP_LOGE(TAG, "Could not initialise rgb888 buffer");
        return;
    }

    uint8_t* display_buf = (uint8_t*)heap_caps_malloc(200 * 200 * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!display_buf) {
        ESP_LOGE(TAG, "Could not initialise display_buf");
        return;
    }

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            printf("Camera capture failed\n");
            return;
        }

        fmt2rgb888((const uint8_t*)fb, 240 * 240 * sizeof(uint16_t), PIXFORMAT_RGB565, rgb888_buf);
        esp_camera_fb_return(fb);

        int8_t *input_ptr = input->data.int8;

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int input_index = (y * 240 + x) * 3;

                uint8_t r = rgb888_buf[input_index];
                uint8_t g = rgb888_buf[input_index + 1];
                uint8_t b = rgb888_buf[input_index + 2];

                *input_ptr++ = r;
                *input_ptr++ = g;
                *input_ptr++ = b;
            }
        }


        if (interpreter->Invoke() != kTfLiteOk) {
            MicroPrintf("Invoke() failed");
        }

        // lv_draw_sw_rgb565_swap(rgb888_buf, 240 * 240);
        float score = dequantize(output->data.int8[0]);
        printf("Score = %f\n", score);

        bsp_display_lock(0);
        if (score < 0.6) { // treat score less than 60% as no person
            lv_led_off(person_indicator);
        } else {
            lv_led_on(person_indicator);
        }

        for (int y = 0; y < 200; y++) {
            memcpy((display_buf + 3 * y * 200), rgb888_buf + 240 * 3 * 20 + y *  240 * 3, 200 * 3);
        }

        lv_canvas_set_buffer(camera_canvas, display_buf, 200, 200, LV_COLOR_FORMAT_RGB888);
        bsp_display_unlock();
    }
}
