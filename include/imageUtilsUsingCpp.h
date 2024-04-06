#include <sycl/sycl.hpp>
#include <array>

extern void ConvertToGrayscaleCpp(
                      const uint8_t *u8_image_in, // input const
                      std::vector<float> &fl_gray, // output
                      int width, int height);

void ConvertToUint8Cpp(
                const std::vector<float> &fl_in, // input. normalized to 0 ... 1
                std::vector<uint8_t> &u8_out,
                int width, int height);