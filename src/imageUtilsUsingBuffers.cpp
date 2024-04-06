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
                      int width, int height)
{
  try
  {  
      q.submit([&fl_grayscale_buffer, &u8_image_in_buffer, width, height](
                sycl::handler& h) {
      auto image = u8_image_in_buffer.get_access<sycl::access::mode::read>(h);
      // A discard_write is a write access that doesn't need to preserve existing
      // memory contents
      auto gray = fl_grayscale_buffer.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for(sycl::range<1>(width * height),
                    [image, gray](sycl::id<1> idx) {
                        int offset   = 3 * idx[0];
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

