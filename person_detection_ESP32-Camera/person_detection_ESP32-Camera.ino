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

#define CONFIG_CAMERA_VFLIP true; 


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
    tft.setRotation(1);
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

  RespondToDetection(error_reporter, person_score, no_person_score);


  camera_fb_t *img = capture();
  
    if (img == nullptr || img == 0) 
    {
        Serial.printf("snap fail\n");
        return;
    }

    //tft.pushImage(160, 160, 320, 480, (uint16_t*)img);
    tft.pushImage(160, 160, img->width, img->height, (lgfx::swap565_t*)img->buf); 

}

  camera_fb_t* capture()
  {
    camera_fb_t *img = NULL;
    esp_err_t res = ESP_OK;
    img = esp_camera_fb_get();
    return img;
  } 

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
    //panel.rgb_order = false;
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

    // Set the value immediately after initializing setRotation. Set the value immediately after initializing setRotation.
    panel.rotation = 0;

    // If you want to change the direction when using setRotation offset_rotation the settings.
    // If you want the direction at setRotation(0) to be the direction at the time of 1, set 1.
    panel.offset_rotation = 0;

    // When you're done, pass the panel pointer in lgfx's setPanel function.
    tft.setPanel(&panel);
}
