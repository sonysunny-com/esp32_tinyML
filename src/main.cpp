/*
 * Project: ESP32-CAM TinyML Person Detection
 * Description:
 *   Runs a TensorFlow Lite Micro model for person detection on the AI Thinker ESP32-CAM.
 *   Captures grayscale frames, resizes to 96×96, performs inference locally,
 *   and toggles the onboard flash LED when a person is detected.
 *
 * Hardware: AI Thinker ESP32-CAM (OV2640)
 * Framework: Arduino (PlatformIO)
 * Author: Sony Sunny
 * Date: 2025-10-22
 */


#include <Arduino.h>          // Core Arduino functions
#include "esp_camera.h"       // ESP32-CAM camera driver

// === Pin definitions for the AI Thinker ESP32-CAM ===
// These map the ESP32 GPIOs to the camera’s physical pins.
#define PWDN_GPIO_NUM     32   // Power down pin (turns camera off/on)
#define RESET_GPIO_NUM    -1   // Reset not used
#define XCLK_GPIO_NUM      0   // XCLK signal for camera clock
#define SIOD_GPIO_NUM     26   // I2C data line to SCCB (camera control bus)
#define SIOC_GPIO_NUM     27   // I2C clock line
#define Y9_GPIO_NUM       35   // Data pin 9
#define Y8_GPIO_NUM       34   // Data pin 8
#define Y7_GPIO_NUM       39   // Data pin 7
#define Y6_GPIO_NUM       36   // Data pin 6
#define Y5_GPIO_NUM       21   // Data pin 5
#define Y4_GPIO_NUM       19   // Data pin 4
#define Y3_GPIO_NUM       18   // Data pin 3
#define Y2_GPIO_NUM        5   // Data pin 2
#define VSYNC_GPIO_NUM    25   // Vertical sync signal
#define HREF_GPIO_NUM     23   // Horizontal reference signal
#define PCLK_GPIO_NUM     22   // Pixel clock signal

// === TensorFlow Lite Micro headers ===
#include "tensorflow/lite/micro/all_ops_resolver.h"  // Registers all supported ops
#include "tensorflow/lite/micro/micro_interpreter.h" // Runs inference on microcontrollers
#include "tensorflow/lite/schema/schema_generated.h" // TFLite model schema
#include "tensorflow/lite/version.h"                 // Version check helper
#include "person_detect_model_data.h"                // Compiled TinyML model (array of bytes)

// === Model input settings ===
static const int kNumCols = 96;       // Model input width
static const int kNumRows = 96;       // Model input height
static const int kNumChannels = 1;    // Grayscale image = 1 channel
static const int kTensorArenaSize = 220 * 1024; // Working memory (RAM) for inference
static uint8_t tensor_arena[kTensorArenaSize];  // Memory buffer used by TFLM

// === TensorFlow model + interpreter pointers ===
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// Flash LED pin on AI Thinker board
const int FLASH_LED_PIN = 4;

// -----------------------------------------------------------------------------
// Function: resize_to_96x96_grayscale
// Downsamples a larger grayscale frame (160x120) to the 96x96 size
// expected by the TinyML model. Uses a simple nearest-neighbor method.
// -----------------------------------------------------------------------------
static bool resize_to_96x96_grayscale(uint8_t* dst, int dw, int dh,
                                      const uint8_t* src, int sw, int sh) {
  if (!dst || !src) return false;          // Sanity check
  int step_x = sw / dw;                    // Horizontal sampling step
  int step_y = sh / dh;                    // Vertical sampling step
  for (int y = 0; y < dh; y++) {
    for (int x = 0; x < dw; x++) {
      dst[y * dw + x] = src[(y * step_y) * sw + (x * step_x)];
    }
  }
  return true;
}

// -----------------------------------------------------------------------------
// Function: init_camera
// Configures and initializes the ESP32-CAM peripheral.
// Returns true if camera setup succeeds.
// -----------------------------------------------------------------------------
static bool init_camera() {
  camera_config_t config = {};             // Initialize configuration struct
  config.ledc_channel = LEDC_CHANNEL_0;    // LEDC timer channel for XCLK PWM
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
  config.xclk_freq_hz = 20000000;          // 20 MHz camera clock
  config.pixel_format = PIXFORMAT_GRAYSCALE; // Capture grayscale images
  config.frame_size   = FRAMESIZE_QQVGA;     // 160×120 resolution
  config.fb_count     = 2;                   // Two frame buffers

  // Initialize the camera driver
  return (esp_camera_init(&config) == ESP_OK);
}

// -----------------------------------------------------------------------------
// setup()
// Runs once at startup.
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);          // Start serial console
  delay(300);
  Serial.println("\n[ESP32-CAM] TinyML Person Detection");

  // Prepare LED (used as output indicator)
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Initialize camera
  if (!init_camera()) {
    Serial.println("Camera init failed");
    while (true) delay(1000);    // Halt here if setup fails
  }

  // Load the compiled TensorFlow Lite model from flash
  model = tflite::GetModel(g_person_detect_model_data);

  // Build interpreter — this binds the model, operations, and tensor arena
  static tflite::AllOpsResolver resolver;  // Includes all operators
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate input/output tensors inside the tensor arena
  interpreter->AllocateTensors();

  // Pointer to input tensor for convenience
  input = interpreter->input(0);
}

// -----------------------------------------------------------------------------
// loop()
// Captures frames continuously, preprocesses them, runs inference,
// and lights the LED if a person is detected.
// -----------------------------------------------------------------------------
void loop() {
  // Capture a frame from the camera
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return; // Skip if capture failed

  // Resize the camera frame (160x120) to 96x96 for the model input
  resize_to_96x96_grayscale(input->data.uint8, kNumCols, kNumRows,
                            fb->buf, fb->width, fb->height);

  // Release the frame buffer so camera can capture next frame
  esp_camera_fb_return(fb);

  // Run inference using TensorFlow Lite Micro
  if (interpreter->Invoke() == kTfLiteOk) {
    // Fetch output tensor (contains model results)
    TfLiteTensor* output = interpreter->output(0);

    // Convert quantized int8 output to floating-point probability
    float person_score = (output->data.int8[1] - output->params.zero_point)
                         * output->params.scale;

    // Print detection confidence to Serial
    Serial.printf("person_score=%.2f\n", person_score);

    // Turn on flash LED if confidence > 0.6
    digitalWrite(FLASH_LED_PIN, person_score > 0.6f ? HIGH : LOW);
  }

  // Small delay before next frame
  delay(200);
}
