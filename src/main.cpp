#include <Arduino.h>
#include "esp_camera.h"

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "person_detect_model_data.h"

static const int kNumCols = 96;
static const int kNumRows = 96;
static const int kNumChannels = 1;
static const int kTensorArenaSize = 220 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

const int FLASH_LED_PIN = 4;

static bool resize_to_96x96_grayscale(uint8_t* dst, int dw, int dh, const uint8_t* src, int sw, int sh) {
  if (!dst || !src) return false;
  int step_x = sw / dw;
  int step_y = sh / dh;
  for (int y = 0; y < dh; y++) {
    for (int x = 0; x < dw; x++) {
      dst[y*dw + x] = src[(y*step_y)*sw + (x*step_x)];
    }
  }
  return true;
}

static bool init_camera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size   = FRAMESIZE_QQVGA;
  config.fb_count     = 2;
  return (esp_camera_init(&config) == ESP_OK);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\n[ESP32-CAM] TinyML Person Detection");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!init_camera()) {
    Serial.println("Camera init failed");
    while (true) delay(1000);
  }

  model = tflite::GetModel(g_person_detect_model_data);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
}

void loop() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return;

  resize_to_96x96_grayscale(input->data.uint8, kNumCols, kNumRows, fb->buf, fb->width, fb->height);
  esp_camera_fb_return(fb);

  if (interpreter->Invoke() == kTfLiteOk) {
    TfLiteTensor* output = interpreter->output(0);
    float person_score = (output->data.int8[1] - output->params.zero_point) * output->params.scale;
    Serial.printf("person_score=%.2f\n", person_score);
    digitalWrite(FLASH_LED_PIN, person_score > 0.6f ? HIGH : LOW);
  }
  delay(200);
}
