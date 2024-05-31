#include <sycl/sycl.hpp>
#include <array>

extern void ConvertToGrayscaleBuffer(sycl::queue &q,
                      sycl::buffer<uint8_t, 1> &u8_image_in_buffer, // input
                      sycl::buffer<float, 1> &fl_grayscale_buffer, // output
                      int width, int height, int numChannels);

extern void ConvertToUint8Buffer(sycl::queue &q, 
                sycl::buffer<float, 1> &fl_in_buffer, // input. normalized to 0 ... 1
                sycl::buffer<uint8_t, 1> &u8_out_buffer,
                int width, int height);

extern void InitUint8SyclBuffer(sycl::queue &q,
                      std::vector<uint8_t> &u8_vector, // input and output
                      int width, int height, uint8_t value);

extern void InitUint8SyclBuffer1(sycl::queue &q,
                      sycl::buffer<uint8_t> &u8_buffer, // input and output
                      //buffer &u8_buffer, // input and output
                      int width, int height, uint8_t value);

extern void SobelFilter(sycl::queue &q,
                 sycl::buffer<float, 1> &fl_in_buffer,
                 sycl::buffer<float, 1> &fl_out_buffer,
                 int width, int height);






