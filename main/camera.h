
#ifndef HUMAN_DETECT_CAMERA_H
#define HUMAN_DETECT_CAMERA_H

#include "sensor.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"

#define CAMERA_PIXEL_FORMAT PIXFORMAT_RGB565
#define CAMERA_FRAME_SIZE FRAMESIZE_240X240

#define CAMERA_MODULE_NAME "ESP-S3-EYE"
#define CAMERA_PIN_PWDN -1
#define CAMERA_PIN_RESET -1

#define CAMERA_PIN_VSYNC 6
#define CAMERA_PIN_HREF 7
#define CAMERA_PIN_PCLK 13
#define CAMERA_PIN_XCLK 15

#define CAMERA_PIN_SIOD 4
#define CAMERA_PIN_SIOC 5

#define CAMERA_PIN_D0 11
#define CAMERA_PIN_D1 9
#define CAMERA_PIN_D2 8
#define CAMERA_PIN_D3 10
#define CAMERA_PIN_D4 12
#define CAMERA_PIN_D5 18
#define CAMERA_PIN_D6 17
#define CAMERA_PIN_D7 16

#define XCLK_FREQ_HZ 15000000

#ifdef __cplusplus
extern "C" {
#endif

int app_camera_init();

#ifdef __cplusplus
}
#endif

#endif