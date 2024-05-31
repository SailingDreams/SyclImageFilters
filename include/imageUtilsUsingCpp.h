#include <sycl/sycl.hpp>
#include <array>

#include <image.h>

float FindMaxCpp(const float *fl_image_in, // input const
                      int width, int height);

void ScaleImgCpp(const float *fl_image_in, // input const
                      float *fl_image_out, // output
                      int width, int height, 
                      float scale);

void ComputeMagnitudeCpp(
                      const float *fl_image_in0, // input const
                      const float *fl_image_in1, // input const
                      std::vector<float> &fl_image_out, // output
                      int width, int height);

extern void ConvertToGrayscaleCpp(
                      const uint8_t *u8_image_in, // input const
                      std::vector<float> &fl_gray, // output
                      int width, int height, int numChannels); // inputs

void ConvertToUint8Cpp(
                const std::vector<float> &fl_in, // input. normalized to 0 ... 1
                std::vector<uint8_t> &u8_out,
                int width, int height);

/****************************************************************************
* Apply the 2D 3x3 filter to the given image.
* @param pOut[out] Output filtered image.
* @param pIn Input image.
* @param pFilter Filter coefficients to apply.  Must be float[9].
* @param border Controls border element processing.
* @param vectorize
* @return Ok if the filter is applied successfully.
*****************************************************************************/
Result Convolution3x3Cpp(float* pOut, const float* pIn, const float* pFilter, 
                      int sx, int sy, int pitch, Border border);
