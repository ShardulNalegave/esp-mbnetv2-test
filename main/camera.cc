
#include "camera.h"
#include "sdkconfig.h"
#include "bsp/esp-bsp.h"

static const char *TAG = "app_camera";

int app_camera_init() {
    bsp_i2c_init();
    camera_config_t config = BSP_CAMERA_DEFAULT_CONFIG;
    config.pixel_format = CAMERA_PIXEL_FORMAT;
    config.frame_size = CAMERA_FRAME_SIZE;

    // camera init
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return -1;
    }
    sensor_t *s = esp_camera_sensor_get();
    s->set_vflip(s, 1); //flip it back
    //initial sensors are flipped vertically and colors are a bit saturated
    if (s->id.PID == OV3660_PID)
    {
        s->set_brightness(s, 1);  //up the blightness just a bit
        s->set_saturation(s, -2); //lower the saturation
    }
    return 0;
}