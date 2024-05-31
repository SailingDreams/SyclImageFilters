#include <cstdio>
#include "imageUtilsAgnostic.h"
#include "imageUtilsUsingBuffers.h"

using namespace sycl;
using namespace std;


/***************************************************************
 * 
****************************************************************/
void ConvertToGrayscaleBuffer(queue &q,
                      buffer<uint8_t, 1> &u8_image_in_buffer, // input
                      buffer<float, 1> &fl_grayscale_buffer, // output
                      int width, int height, int numChannels)
{
  try
  {  
      q.submit([&fl_grayscale_buffer, &u8_image_in_buffer, width, height, numChannels](
                sycl::handler& h) {
      auto image = u8_image_in_buffer.get_access<sycl::access::mode::read>(h);
      // A discard_write is a write access that doesn't need to preserve existing
      // memory contents
      auto gray = fl_grayscale_buffer.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for(sycl::range<1>(width * height),
                    [image, gray, numChannels](sycl::id<1> idx) {
                        int offset   = numChannels * idx[0];
                        gray[idx[0]] = luminance(image[offset], image[offset + 1],
                                                 image[offset + 2]);
                    });
      });
  } catch (std::exception const &e) {
    cout << "convertToGrayscale exception: " << e.what() << std::endl;
    terminate();
  }
}

/***************************************************************
 * 
****************************************************************/
void ConvertToUint8Buffer(queue &q, 
                buffer<float, 1> &fl_in_buffer, // input. normalized to 0 ... 1
                buffer<uint8_t, 1> &u8_out_buffer,
                int width, int height)
{
  try
  {  
      range num_items(width * height);
      //q.submit([&fl_in_buffer, &u8_out_buffer, width, height](
      //          sycl::handler& h) 
      q.submit([&](auto &h)                 
      {
        // A discard_write is a write access that doesn't need to preserve existing
        // memory contents
        auto fl_in = fl_in_buffer.get_access<sycl::access::mode::read>(h);
        auto u8_out = u8_out_buffer.get_access<sycl::access::mode::write>(h);

        //h.parallel_for(sycl::range<1>(width * height),
        h.parallel_for(num_items,
                      //[fl_in, u8_out](sycl::id<1> idx) {
                      [=](auto idx) {
                          //u8_out[idx[0]] = fl_in[idx[0]] * 255;
                          u8_out[idx] = fl_in[idx] * 255;
                      });
      });
  } catch (std::exception const &e) {
    cout << "convertToGrayscale exception: " << e.what() << std::endl;
    terminate();
  }
}

/***************************************************************
 * Sycl buffer not passed as argument. The vector arg mem will be 
 * sync'd with host when this function returns.
 ****************************************************************/
void InitUint8SyclBuffer(queue &q,
                      vector<uint8_t> &u8_vector, // input and output
                      int width, int height, uint8_t value)
{
  range numItems{width * height};

  buffer u8_buffer(u8_vector);
  try
  {  
      //q.submit([&u8_buffer, width, height, value](
      //          sycl::handler& h) {
      q.submit([&](auto &h) {
      // A discard_write is a write access that doesn't need to preserve existing
      // memory contents
      //auto u8 = u8_buffer.get_access<sycl::access::mode::write>(h);
      accessor u8(u8_buffer, h, write_only, no_init);

      //h.parallel_for(sycl::range<1>(width * height),
      //              [u8, value](sycl::id<1> idx) {
      //                  u8[idx[0]] = value;
      //              });
      h.parallel_for(numItems, [=](auto idx) {
                        u8[idx] = value;
                    });
      });
  } catch (std::exception const &e) {
    cout << "initUint8SyclBuffer exception: " << e.what() << std::endl;
    terminate();
  }
}

/***************************************************************
 * Used for cases where you don't want the sycl buffer to go out of scope
 ****************************************************************/
void InitUint8SyclBuffer1(queue &q,
                      buffer<uint8_t> &u8_buffer, // input and output
                      //buffer &u8_buffer, // input and output
                      int width, int height, uint8_t value)
{
  range numItems{width * height};

  try
  {  
      //q.submit([&u8_buffer, width, height, value](
      //          sycl::handler& h) {
      q.submit([&](auto &h) {
      // A discard_write is a write access that doesn't need to preserve existing
      // memory contents
      //auto u8 = u8_buffer.get_access<sycl::access::mode::write>(h);
      accessor u8(u8_buffer, h, write_only, no_init);

      //h.parallel_for(sycl::range<1>(width * height),
      //              [u8, value](sycl::id<1> idx) {
      //                  u8[idx[0]] = value;
      //              });
      h.parallel_for(numItems, [=](auto idx) {
                        u8[idx] = value;
                    });
      });
  } catch (std::exception const &e) {
    cout << "initUint8SyclBuffer exception: " << e.what() << std::endl;
    terminate();
  }
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
 * https://www.codeproject.com/Articles/5284847/5-Minutes-to-Your-First-oneAPI-App-on-DevCloud
****************************************************************/
void SobelFilter(sycl::queue &queue,
                 sycl::buffer<float, 1> &fl_in_buffer, // a grayscale buffer with 1 channel
                 sycl::buffer<float, 1> &fl_out_buffer,
                 int width, int height)
{
  sycl::buffer<float, 1> dx{width * height};  // todo make these device only memory
  sycl::buffer<float, 1> dy{width * height};

  // the horizontal convolution
  {
    // Open a new scope so that dx_tmp is deallocated once no longer needed
    sycl::buffer<float, 1> dx_tmp{width * height};

    // Extract a 3x1 window around (x, y) and compute the dot product
    // between the window and the kernel [1, 0, -1]
    queue.submit([&fl_in_buffer, &dx_tmp, width, height](sycl::handler& h)
    {
      auto data = fl_in_buffer.get_access<sycl::access::mode::read>(h);
      auto out = dx_tmp.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for(sycl::range<2>(width, height),
                      [data, width, out](sycl::id<2> idx) {
                          int offset = idx[1] * width + idx[0];
                          float left = idx[0] == 0 ? 0 : data[offset - 1];
                          float right = idx[0] == width - 1 ? 0 : data[offset + 1];
                          out[offset] = left - right;
                      });
    });

    // Extract a 1x3 window around (x, y) and compute the dot product
    // between the window and the kernel [1, 2, 1]
    queue.submit([&dx, &dx_tmp, width, height](sycl::handler& h) 
    {
      auto data = dx_tmp.get_access<sycl::access::mode::read>(h);
      auto out  = dx.get_access<sycl::access::mode::discard_write>(h);
      h.parallel_for(
            sycl::range<2>(width, height),
            [data, width, height, out](sycl::id<2> idx) {
                // Convolve vertically
                int offset = idx[1] * width + idx[0];
                float up   = idx[1] == 0 ? 0 : data[offset - width];
                float down = idx[1] == height - 1 ? 0 : data[offset + width];
                float center = data[offset];
                out[offset]  = up + 2 * center + down;
            });
    });
  }
  //#if 0
  // The vertical convolution is then performed in the same way, except with different kernels:
  {
    sycl::buffer<float, 1> dy_tmp{width * height};

    queue.submit([&fl_in_buffer, &dy_tmp, width, height](
                 sycl::handler& h) 
    {
      auto data = fl_in_buffer.get_access<sycl::access::mode::read>(h);
      auto out  = dy_tmp.get_access<sycl::access::mode::discard_write>(h);

      // Create a scratch buffer for the intermediate computation
      h.parallel_for(sycl::range<2>(width, height),
                    [data, width, out](sycl::id<2> idx) {
                        // Convolve horizontally
                        int offset = idx[1] * width + idx[0];
                        float left = idx[0] == 0 ? 0 : data[offset - 1];
                        float right = idx[0] == width - 1 ? 0 : data[offset + 1];
                        float center = data[offset];
                        out[offset]  = left + 2 * center + right;
                      });
    });

    queue.submit([&dy, &dy_tmp, width, height](sycl::handler& h) 
    {
      auto data = dy_tmp.get_access<sycl::access::mode::read>(h);
      auto out  = dy.get_access<sycl::access::mode::discard_write>(h);
      h.parallel_for(
          sycl::range<2>(width, height),
          [data, width, height, out](sycl::id<2> idx) {
              // Convolve vertically
              int offset = idx[1] * width + idx[0];
              float up   = idx[1] == 0 ? 0 : data[offset - width];
              float down = idx[1] == height - 1 ? 0 : data[offset + width];
              out[offset] = up - down;
          });
    });
  }
  #if 1
  
  // Notice that the above vertical and horizontal gradients have no dependence 
  // on one another, so SYCL may execute them in parallel.

  // For each pixel, we can have the gradient projected on the x and y axes, 
  // so it's a simple matter to compute the magnitude of the gradient.
  queue.submit([&dx, &dy, width, height, &fl_out_buffer](sycl::handler& h) {
      auto dx_data = dx.get_access<sycl::access::mode::read>(h);
      auto dy_data = dy.get_access<sycl::access::mode::read>(h);
      auto fl_out = fl_out_buffer.get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::range<1>(width * height),
          [dx_data, dy_data, fl_out](sycl::id<1> idx) {
              float dx_val = dx_data[idx[0]];
              float dy_val = dy_data[idx[0]];
              // NOTE: if deploying to an accelerated device, math
              // functions MUST be used from the sycl namespace
              fl_out[idx[0]] = sycl::sqrt(dx_val * dx_val + dy_val * dy_val);
      });
  });
  
  #endif
}
