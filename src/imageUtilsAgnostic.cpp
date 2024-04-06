#include "imageUtilsAgnostic.h"

/*************************************************
 Convert rbb to gray scale
*/
float luminance(uint8_t r, uint8_t g, uint8_t b)
{
    float r_lin = static_cast<float>(r) / 255.0f;
    float g_lin = static_cast<float>(g) / 255.0f;
    float b_lin = static_cast<float>(b) / 255.0f;

    // Perceptual luminance (CIE 1931)
    return 0.2126f * r_lin + 0.7152f * g_lin + 0.0722f * b_lin;
    //return r_lin;
    //return 0.5;
}