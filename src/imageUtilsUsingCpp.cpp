#include <cstdio>
#include <cmath>
#include "imageUtilsAgnostic.h"
#include "imageUtilsUsingCpp.h"

using namespace std;

/***************************************************************
 * 
 ****************************************************************/
float FindMaxCpp(const float *fl_image_in, // input const
                      int width, int height)
{
    float maxVal = 0.0f;
    for(int idx = 0; idx < (width * height); idx++)
    {        
        maxVal = std::max(fl_image_in[idx], maxVal);
    }
    return maxVal;  
}

/***************************************************************
 * 
 ****************************************************************/
void ScaleImgCpp(
                      const float *fl_image_in, // input const
                            float *fl_image_out, // output
                      int width, int height, 
                      float scale)
{
    for(int idx = 0; idx < (width * height); idx++)
    {
        float i0 = fl_image_in[idx];
        fl_image_out[idx] = i0 * scale;
    }  
}

/***************************************************************
 * 
 ****************************************************************/
void ComputeMagnitudeCpp(
                      const float *fl_image_in0, // input const
                      const float *fl_image_in1, // input const
                      vector<float> &fl_image_out, // output
                      int width, int height)
{
    for(int idx = 0; idx < (width * height); idx++)
    {
        float i0 = fl_image_in0[idx];
        float i1 = fl_image_in1[idx];
        fl_image_out[idx] = sqrtf(i0 * i0 + i1 * i1);
    }  
}

/***************************************************************
 * 
 ****************************************************************/
void ConvertToGrayscaleCpp(
                      const uint8_t *u8_image_in, // input const
                      vector<float> &fl_gray, // output
                      int width, int height, int numChannels)
{
  try
  {  
      for(int idx = 0; idx < (width * height); idx++)
      {
        int offset   = numChannels * idx;
        fl_gray[idx] = luminance(u8_image_in[offset],
            u8_image_in[offset + 1],
            u8_image_in[offset + 2]);
      }
  } catch (std::exception const &e) {
    cout << "convertToGrayscale exception: " << e.what() << std::endl;
    terminate();
  }
}

/***************************************************************
 * 
 ****************************************************************/
void ConvertToUint8Cpp(
                const vector<float> &fl_in, // input. normalized to 0 ... 1
                vector<uint8_t> &u8_out,
                int width, int height)
{
  try
  {  
    for (int idx = 0; idx < width * height; idx++)
    {
      u8_out[idx] = static_cast<uint8_t>(std::max(0.0f, fl_in[idx] * 255.0f));  // consider negative img values
    }
  } catch (std::exception const &e) {
    cout << "convertToUint8Cpp exception: " << e.what() << std::endl;
    terminate();
  }
}
/***************************************************************
 * 
 ****************************************************************/
Result Convolution3x3Cpp(float* pOut, const float* pIn, const float* pFilter, 
                      int sx, int sy, int pitch, Border border)
{
    if (pOut == nullptr || pIn == nullptr || pFilter == nullptr) return InvalidArgument;

    for (int y = 0; y < sy; y++)
    {
        for (int x = 0; x < sx; x++)
        {
            float value = 0.0f;
            int cIdx = 0;
            switch (border)
            {
            case Border::Clamp:
                for (int l = -1; l <= 1; l++)  // filter col
                {
                    for (int k = -1; k <= 1; k++)  // filter row
                    {
                        //float v = pIn->atClamp(x + k, y + l);
                        int x1 = std::max(0, std::min(x + k, sx - 1));
                        int y1 = std::max(0, std::min(y + l, sy - 1));
                        float v = pIn[y1 * pitch + x1];
                        float c = pFilter[cIdx++];
                        value += v * c;
                        //value = pIn[y1 * pitch + x1];
                    }
                }
                break;
#if 0
            case Border::Wrap:
                for (int l = -1; l <= 1; l++)
                {
                    for (int k = -1; k <= 1; k++)
                    {
                        float v = pIn->atWrap(x + k, y + l);
                        float c = pFilter[cIdx++];
                        value += v * c;
                    }
                }
                break;
            case Border::Reflect:
                for (int l = -1; l <= 1; l++)
                {
                    for (int k = -1; k <= 1; k++)
                    {
                        float v = pIn->atReflect(x + k, y + l);
                        float c = pFilter[cIdx++];
                        value += v * c;
                    }
                }
                break;
            case Border::Mirror:
                for (int l = -1; l <= 1; l++)
                {
                    for (int k = -1; k <= 1; k++)
                    {
                        float v = pIn->atMirror(x + k, y + l);
                        float c = pFilter[cIdx++];
                        value += v * c;
                    }
                }
                break;
#endif
            default:return Result::InvalidArgument;
            }
            pOut[y*pitch+x] = value;
        }
    }
return Result::Ok;
}

/***************************************************************
 * Sobel Filter implemented using horizontal and vertical convolutions
 * |1  0 -1|
 * |2  0 -2|
 * |1  0 -1|
 * 
 * | 1  2  1|
 * | 0  0  0|
 * |-1 -2 -1|
 * 
 * Seperable version of the above filter
 * |1  0 -1|   |1|
 * |2  0 -2| = |2| * [1  0 -1]
 * |1  0 -1|   |1| 
 * 
 * Ref: 
 * 
****************************************************************/
void SobelFilterCpp(std::vector<float> &fl_in_buffer, // a grayscale buffer with 1 channel
                 std::vector<float> &fl_out_buffer,
                 int width, int height)
{
    vector<float> sobelXGradient(width * height);
    vector<float> sobelYGradient(width * height);
    //float sobelXFilter[9] = {0, 0, 0,  
    //                   0, 1, 0, 
    //                   0, 0,  0};
    float sobelXFilter[9] = {1, 0, -1,  
                        2, 0, -2, 
                        1, 0,  -1};
    float sobelYFilter[9] = {1, 2,   1,  
                        0, 0,   0, 
                        -1, -2, -1};

    // Convolve the x gradient
    Convolution3x3Cpp(sobelXGradient.data(), fl_in_buffer.data(), &sobelXFilter[0], width, height, width, Border::Clamp);
    // Convolve the y gradient
    Convolution3x3Cpp(sobelYGradient.data(), fl_in_buffer.data(), &sobelYFilter[0], width, height, width, Border::Clamp);
    // Find max of both gradients
    float maxValX = FindMaxCpp(sobelXGradient.data(), width, height);
    //cout << "Max SobelX value = " << maxValX << std::endl;
    float maxValY = FindMaxCpp(sobelYGradient.data(), width, height);
    //cout << "Max SobelY value = " << maxValY << std::endl;
    // float maxValXY = std::max(maxValX, maxValY); // todo: look up fix in tracker.h to get template version of max()
    
    float maxValXY = maxValX < maxValY ? maxValY : maxValX;

    // Normalize both X and Y
    vector<float> sobelXScaled(width * height);
    ScaleImgCpp(sobelXGradient.data(), sobelXScaled.data(), width, height, 1.0f/maxValXY);
    vector<float> sobelYScaled(width * height);
    ScaleImgCpp(sobelYGradient.data(), sobelYScaled.data(), width, height, 1.0f/maxValXY);

    //maxVal = FindMaxCpp(sobelXScaled.data(), width, height); // debug
    //cout << "Max scaled image value = " << maxVal << std::endl;  // debug
    
#if 0
    // Convert to uint8
    std::vector<uint8_t> u8_imageX_out(width * height);
    ConvertToUint8Cpp(sobelXScaled, u8_imageX_out, width, height);
    stbi_write_png("image_sobelX.png", width, height, 1, u8_imageX_out.data(), width);  
    std::vector<uint8_t> u8_imageY_out(width * height);
    ConvertToUint8Cpp(sobelYScaled, u8_imageY_out, width, height);
    stbi_write_png("image_sobelY.png", width, height, 1, u8_imageY_out.data(), width);
#endif

    // Compute magnitude (final step of Sobel)
    ComputeMagnitudeCpp(sobelXScaled.data(), sobelYScaled.data(), fl_out_buffer, width, height);
}
