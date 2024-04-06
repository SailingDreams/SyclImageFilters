#include <cstdio>
#include "imageUtilsAgnostic.h"
#include "imageUtilsUsingCpp.h"

using namespace std;
/***************************************************************
 * 
 ****************************************************************/
void ConvertToGrayscaleCpp(
                      const uint8_t *u8_image_in, // input const
                      vector<float> &fl_gray, // output
                      int width, int height)
{
  try
  {  
      for(int idx = 0; idx < (width * height); idx++)
      {
        int offset   = 3 * idx;
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
      u8_out[idx] = static_cast<uint8_t>(fl_in[idx] * 255.0f);
    }
  } catch (std::exception const &e) {
    cout << "convertToUint8Cpp exception: " << e.what() << std::endl;
    terminate();
  }
}