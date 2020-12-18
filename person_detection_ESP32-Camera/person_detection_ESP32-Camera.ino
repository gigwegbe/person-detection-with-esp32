/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <Wire.h>
#include "SPI.h"
#include <LovyanGFX.hpp>
#include "makerfabs_pin.h"
#include "esp_camera.h"

//#include <TJpg_Decoder.h>

//Choice your touch IC
//#define NS2009_TOUCH
#define FT6236_TOUCH

#ifdef NS2009_TOUCH
#include "NS2009.h"
const int i2c_touch_addr = NS2009_ADDR;
#endif


#ifdef FT6236_TOUCH
#include "FT6236.h"
const int i2c_touch_addr = TOUCH_I2C_ADD;
#endif

//SPI control
#define SPI_ON_TFT digitalWrite(LCD_CS, LOW)
#define SPI_OFF_TFT digitalWrite(LCD_CS, HIGH)


struct LGFX_Config
{
    static constexpr spi_host_device_t spi_host = VSPI_HOST;
    static constexpr int dma_channel = 1;
    static constexpr int spi_sclk = LCD_SCK;
    static constexpr int spi_mosi = LCD_MOSI;
    static constexpr int spi_miso = LCD_MISO;
};

static lgfx::LGFX_SPI<LGFX_Config> tft;
static LGFX_Sprite sprite(&tft);
static lgfx::Panel_ILI9488 panel;

int last_pos[2] = {0, 0};
int draw_color = TFT_WHITE;


#include <TensorFlowLite_ESP32.h>

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


uint8_t red[635] = {0xff,0xd8,0xff,0xe0,0x0,0x10,0x4a,0x46,0x49,0x46,0x0,0x1,0x1,0x0,0x0,0x1,0x0,0x1,0x0,0x0,0xff,
  0xdb,0x0,0x43,0x0,0x2,0x1,0x1,0x1,0x1,0x1,0x2,0x1,0x1,0x1,0x2,0x2,0x2,0x2,0x2,0x4,0x3,0x2,0x2,0x2,0x2,0x5,0x4,0x4,
  0x3,0x4,0x6,0x5,0x6,0x6,0x6,0x5,0x6,0x6,0x6,0x7,0x9,0x8,0x6,0x7,0x9,0x7,0x6,0x6,0x8,0xb,0x8,0x9,0xa,0xa,0xa,0xa,0xa,
  0x6,0x8,0xb,0xc,0xb,0xa,0xc,0x9,0xa,0xa,0xa,0xff,0xdb,0x0,0x43,0x1,0x2,0x2,0x2,0x2,0x2,0x2,0x5,0x3,0x3,0x5,0xa,0x7,0x6,
  0x7,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,
  0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xff,0xc0,0x0,0x11,0x8,0x0,0x8,0x0,0x8,
  0x3,0x1,0x22,0x0,0x2,0x11,0x1,0x3,0x11,0x1,0xff,0xc4,0x0,0x1f,0x0,0x0,0x1,0x5,0x1,0x1,0x1,0x1,0x1,0x1,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xff,0xc4,0x0,0xb5,0x10,0x0,0x2,0x1,0x3,0x3,0x2,0x4,0x3,0x5,0x5,0x4,0x4,0x0,
  0x0,0x1,0x7d,0x1,0x2,0x3,0x0,0x4,0x11,0x5,0x12,0x21,0x31,0x41,0x6,0x13,0x51,0x61,0x7,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x8,0x23,
  0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x9,0xa,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,0x29,0x2a,0x34,0x35,
  0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
  0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,
  0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,
  0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,
  0xff,0xc4,0x0,0x1f,0x1,0x0,0x3,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xff,
  0xc4,0x0,0xb5,0x11,0x0,0x2,0x1,0x2,0x4,0x4,0x3,0x4,0x7,0x5,0x4,0x4,0x0,0x1,0x2,0x77,0x0,0x1,0x2,0x3,0x11,0x4,0x5,0x21,0x31,0x6,0x12,0x41,0x51,
  0x7,0x61,0x71,0x13,0x22,0x32,0x81,0x8,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x9,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0xa,0x16,0x24,0x34,0xe1,0x25,
  0xf1,0x17,0x18,0x19,0x1a,0x26,0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,
  0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,
  0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,
  0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,
  0xf7,0xf8,0xf9,0xfa,0xff,0xda,0x0,0xc,0x3,0x1,0x0,0x2,0x11,0x3,0x11,0x0,0x3f,0x0,0xf8,0xbe,0x8a,0x28,0xaf,0xe5,0x33,0xfd,0xfc,0x3f,0xff,0xd9};

  uint8_t white[631] = {0xff,0xd8,0xff,0xe0,0x0,0x10,0x4a,0x46,0x49,0x46,0x0,0x1,0x1,0x0,0x0,0x1,0x0,0x1,0x0,0x0,0xff,0xdb,0x0,0x43,0x0,0x2,0x1,0x1,
  0x1,0x1,0x1,0x2,0x1,0x1,0x1,0x2,0x2,0x2,0x2,0x2,0x4,0x3,0x2,0x2,0x2,0x2,0x5,0x4,0x4,0x3,0x4,0x6,0x5,0x6,0x6,0x6,0x5,0x6,0x6,0x6,0x7,0x9,0x8,0x6,0x7,
  0x9,0x7,0x6,0x6,0x8,0xb,0x8,0x9,0xa,0xa,0xa,0xa,0xa,0x6,0x8,0xb,0xc,0xb,0xa,0xc,0x9,0xa,0xa,0xa,0xff,0xdb,0x0,0x43,0x1,0x2,0x2,0x2,0x2,0x2,0x2,0x5,
  0x3,0x3,0x5,0xa,0x7,0x6,0x7,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,
  0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xff,0xc0,0x0,0x11,0x8,0x0,0x8,0x0,0x8,0x3,0x1,0x22,0x0,0x2,0x11,0x1,
  0x3,0x11,0x1,0xff,0xc4,0x0,0x1f,0x0,0x0,0x1,0x5,0x1,0x1,0x1,0x1,0x1,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xff,
  0xc4,0x0,0xb5,0x10,0x0,0x2,0x1,0x3,0x3,0x2,0x4,0x3,0x5,0x5,0x4,0x4,0x0,0x0,0x1,0x7d,0x1,0x2,0x3,0x0,0x4,0x11,0x5,0x12,0x21,0x31,0x41,0x6,0x13,0x51,0x61,
  0x7,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x8,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x9,0xa,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,
  0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,
  0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,
  0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,
  0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xc4,0x0,0x1f,0x1,
  0x0,0x3,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xa,0xb,0xff,0xc4,0x0,0xb5,0x11,0x0,0x2,0x1,0x2,
  0x4,0x4,0x3,0x4,0x7,0x5,0x4,0x4,0x0,0x1,0x2,0x77,0x0,0x1,0x2,0x3,0x11,0x4,0x5,0x21,0x31,0x6,0x12,0x41,0x51,0x7,0x61,0x71,0x13,0x22,0x32,0x81,0x8,0x14,
  0x42,0x91,0xa1,0xb1,0xc1,0x9,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0xa,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,0x27,0x28,0x29,0x2a,0x35,
  0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,
  0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,
  0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe2,0xe3,
  0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xda,0x0,0xc,0x3,0x1,0x0,0x2,0x11,0x3,0x11,0x0,0x3f,0x0,0xfd,0xfc,0xa2,
  0x8a,0x28,0x3,0xff,0xd9};



// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 70 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  while (!Serial)
        ; // Leonardo: wait for serial monitor
    Serial.println("\n NS2009 test");

    Wire.begin(I2C_SDA, I2C_SCL);
    byte error, address;

    Wire.beginTransmission(i2c_touch_addr);
    error = Wire.endTransmission();

    if (error == 0)
    {
        Serial.print("I2C device found at address 0x");
        Serial.print(i2c_touch_addr, HEX);
        Serial.println("  !");
    }
    else if (error == 4)
    {
        Serial.print("Unknown error at address 0x");
        Serial.println(i2c_touch_addr, HEX);
    }

    pinMode(LCD_CS, OUTPUT);
    SPI_OFF_TFT;
    SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);

  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                       tflite::ops::micro::Register_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_AVERAGE_POOL_2D,
      tflite::ops::micro::Register_AVERAGE_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  //TFT(SPI) init
    SPI_ON_TFT;

    set_tft();
    tft.begin();
    //tft.init();
    //tft.fillScreen(TFT_RED);
    SPI_OFF_TFT;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.uint8)) {
    error_reporter->Report("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    error_reporter->Report("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];


  //showingImage();
  RespondToDetection(error_reporter, person_score, no_person_score);


  camera_fb_t *img = capture();
  //uint8_t *img = camera.snapshot();
    if (img == nullptr || img == 0) {
        Serial.printf("snap fail\n");
        return;
    }

    //tft.pushImage(0, 0, 320, 480, (uint16_t*)img);
    
    tft.drawPng(( std :: uint8_t *)img, 731,0 ,0);
  
}
 camera_fb_t* capture(){
  camera_fb_t *img = NULL;
  esp_err_t res = ESP_OK;
  img = esp_camera_fb_get();
  return img;
}

//  void showingImage() {
//   camera_fb_t *fb = capture();
//   if (!fb || fb->format != PIXFORMAT_JPEG) {
//     Serial.println("Camera capture failed");
//     esp_camera_fb_return(fb);
//     return;
//   } else {
//     //TJpgDec.drawJpg(0, 0, (const uint8_t*)fb->buf, fb->len);
//     tft.pushImage(0, 0, 198, 198, (uint16_t*)fb);
//     esp_camera_fb_return(fb);
//   }
// }


void set_tft()
{
    // I will add various settings to the panel class.
    // (If you select a panel class for an LCD integrated product,
    //   The initial value is set to match the product, so no setting is required.)

    // Set the SPI clock during normal operation.
    // Esp32 SPI is only available with 80MHz broken by an integer.
    // The configurable value closest to the set value is used.
    panel.freq_write = 60000000;
    //panel.freq_write = 20000000;

    // Set the SPI clock during the single-color fill process.
    // Basically freq_write same value as the same value.
    // Setting a higher value may still work.
    panel.freq_fill = 60000000;
    //panel.freq_fill  = 27000000;

    // Set the SPI clock for reading pixel data from the LCD.
    panel.freq_read = 16000000;

    // Set spi communication mode from 0 to 3.
    panel.spi_mode = 0;

    // Set spi communication mode at the time of data reading from 0 to 3.
    panel.spi_mode_read = 0;

    // Set the number of dummy bits when reading pixels.
    //  Adjust when bit shift occurs in pixel reading.
    panel.len_dummy_read_pixel = 8;

    // Set true for panels that can read data, false if not possible.
    // If omitted, it will be true.
    panel.spi_read = true;

    // Set true for panels that are performed with the read data MOSI pin.
    // If omitted, it will be false.
    panel.spi_3wire = false;

    // Set the pin number to which the LCD CS is connected.
    // If not, omit it or set -1.
    panel.spi_cs = LCD_CS;

    // Set the pin number to which the LCD DC is connected.
    panel.spi_dc = LCD_DC;

    //  Set the pin number to which the LCD RST is connected.
    // If not, omit it or set -1.
    panel.gpio_rst = LCD_RST;

    // Set the pin number to which the LCD backlight is connected.
    //  If not, omit it or set -1.
    panel.gpio_bl = LCD_BL;

    // When using backlight, set the PWM channel number used for brightness control.
    // If you do not want to use PWM brightness control, omit it or set -1.
    panel.pwm_ch_bl = -1;

    //  Set the output level when the backlight is on low or high.
    // True when omitted. It lights up at true=HIGH / false=LOW.
    panel.backlight_level = true;

    // Set the initial value of invertDisplay.If true, it will be reversed.
    // False when omitted.If the color of the screen is inverted, change the settings.
    panel.invert = false;

    // RGB=true / BGR=false The color order of the panels is set.RGB=true / BGR=false
    // False when omitted.If red and blue are replaced, change the settings.
    panel.rgb_order = false;

    // Set the number of pixels (width and height) that the panel's memory has.
    // If the settings are not right, the coordinates when using setRotation are off.
    // (Example: ST7735 has three streets: 132x162 / 128x160 / 132x132)
    panel.memory_width = LCD_WIDTH;
    panel.memory_height = LCD_HEIGHT;

    // Set the actual number of pixels (width and height) of the panel.
    // When omitted, the default value of the panel class is used.
    panel.panel_width = LCD_WIDTH;
    panel.panel_height = LCD_HEIGHT;

    // Set the amount of offset for the panel.
    // When omitted, the default value of the panel class is used.
    panel.offset_x = 0;
    panel.offset_y = 0;

    // Set the value immediately after initializing setRotation.Set the value immediately after initializing setRotation.
    panel.rotation = 0;

    // If you want to change the direction when using setRotation offset_rotation the settings.
    // If you want the direction at setRotation(0) to be the direction at the time of 1, set 1.
    panel.offset_rotation = 0;

    // When you're done, pass the panel pointer in lgfx's setPanel function.
    tft.setPanel(&panel);
}
