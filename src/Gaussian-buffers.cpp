//==============================================================
// Iota is the equivalent of a Hello, World! sample for data parallel programs.
// Building and running the sample verifies that your development environment
// is setup correctly and demonstrates the use of the core features of SYCL.
// This sample runs on both CPU and GPU (or FPGA). When run, it computes on both
// the CPU and offload device, then compares results. If the code executes on
// both CPU and the offload device, the name of the offload device and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <windows.h> 

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//#pragma comment(linker, "/HEAP:10000000")  // This is ignored

//#include <oneapi/ipl/filter/gaussian.hpp>

#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
  #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;
using namespace std;

const int EXIT_ERROR_CODE = 1;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

// Array type and data size for this example.
constexpr size_t array_size = 10000;
typedef array<int, array_size> IntArray;

/*************************************************
 Conver rbb to gray scale
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

/***************************************************************
 * 
 ****************************************************************/
void convertToGrayscaleCpp(
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
//void initUint8SyclBuffer(queue &q,
//                      buffer<uint8_t, 1> &u8_buffer, // input and output
//                      int width, int height, uint8_t value)
void initUint8SyclBuffer(queue &q,
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
 * 
 ****************************************************************/
void initUint8SyclBuffer1(queue &q,
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
 * 
 ****************************************************************/
void convertToGrayscale(queue &q,
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
                        gray[idx[0]] = luminance(image[offset],
                        image[offset + 1],
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
void convertToUint8Cpp(
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

/***************************************************************
 * 
****************************************************************/
void convertToUint8(queue &q, 
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

//**************************************************************************
// Demonstrate iota both sequential on CPU and parallel on device.
//**************************************************************************
int main() {
  // Create device selector for the device of your interest.
  // The default device selector will select the most performant device.
  auto selector = default_selector_v;
  cout << "Starting main" << std::endl;

  int channels;
  int width; 
  int height; 
  string path = "..\\images\\peppers.png";
  const int LOAD_IMAGE_AS_IS = 0;

  #if 0
  // Get a handle to the process heap.
  HANDLE hHeap = GetProcessHeap();
  cout << "Got heap handle = " << hHeap << std::endl;
  
  SIZE_T dwHeapSize = 0;

  try {
    // Get the size of the heap.
    dwHeapSize = HeapSize(hHeap, 0, NULL);
    cout << "Got heap handle" << std::endl;
  }
  catch(const std::exception &exc) {
    std::cerr << exc.what();
  }

  // Print the heap size to the console.
  cout << "The heap size is " << dwHeapSize << " bytes.\n";
  #endif

  #if 0
  struct mallinfo mi;
  mi = std::mallinfo();
  cout << "The heap size is: " << mi.arena/(1024) << " kBytes." << std::endl;
  cout << "The heap size is: " << mi.arena/(1024 * 1024) << " MBytes." << std::endl;
  #endif


  uint8_t* u8_image_in = stbi_load(path.c_str(), &width, &height, &channels, LOAD_IMAGE_AS_IS);
  if (u8_image_in == nullptr) 
  {
    cout << "ERROR: could not load image " << path << std::endl;
    exit(EXIT_ERROR_CODE);
  }
  cout << "Loaded image " << path << " of width = " << width << ", height = " << height << ", num channels = " << channels << std::endl;
  stbi_write_png("image_as_read_in_3chan.png", width, height, channels, u8_image_in, width * channels);
  
  //uint8_t* u8_out = reinterpret_cast<uint8_t*>(sycl::malloc_shared(width * height, sycl_que));
  std::vector<uint8_t> u8_image_out(width * height);
  //array<uint8_t, 512*512> u8_image_out;
  cout << "u8_image_out size " << u8_image_out.size() << std::endl;  

  // Target to hold grayscale image. This constructor indicates that the memory should be allocated by the runtime
  vector<float> fl_grayscale(width * height);

  { // Set scope for SYCL buffers

    // Create sycl buffer for the input image
    buffer<uint8_t, 1> u8_image_in_buffer{u8_image_in, width * height * channels};
    // Create sycl buffer for the grayscale image
    buffer<float, 1> fl_grayscale_buffer{fl_grayscale.data(),width * height};

    //buffer<uint8_t, 1> u8_image_out_buffer{u8_image_out.data(), width * height};
    buffer u8_image_out_buffer(u8_image_out);

    #define USE_SYCL
    //#undef USE_SYCL

    queue sycl_que(selector, exception_handler);

    try {
      //queue q(selector, exception_handler);

      // Print out the device information used for the kernel code.
      cout << "Running on device: "
          << sycl_que.get_device().get_info<info::device::name>() << "\n";
      
      #ifdef USE_SYCL
        cout << "Using SYCL" << std::endl;
        convertToGrayscale(sycl_que, u8_image_in_buffer, fl_grayscale_buffer, width, height);

        convertToUint8(sycl_que, fl_grayscale_buffer, u8_image_out_buffer, width, height);
        //initUint8SyclBuffer(sycl_que, u8_image_out, width, height, (uint8_t)128);
        //initUint8SyclBuffer1(sycl_que, u8_image_out_buffer, width, height, (uint8_t)128);

      #endif

    } catch (std::exception const &e) {
      cout << "An exception is caught while computing IotaParallel on device:  " << e.what() << std::endl;
      terminate();
    }
    sycl_que.wait();
  } // End scope for SYCL buffers - causes synchronization with host.
  
  #ifndef USE_SYCL
    this line is here to create a compile error if USE_SYCL is undefined
    cout << "Running C++ version" << std::endl;
    convertToGrayscaleCpp(u8_image_in, fl_grayscale, width, height);

    convertToUint8Cpp(fl_grayscale, u8_image_out, width, height);
    cout << "Done running C++" << std::endl;
  #endif

  // Print out some image values
  cout << "Greyscale image in float" << std::endl;
  for(int i = 0; i < 10; i++) { cout << fl_grayscale[i] << ", ";}
  cout << std::endl;
  cout << "Greyscale image in uint8" << std::endl;
  for(int i = 0; i < 10; i++) { cout << (int)u8_image_out[i] << ", ";}
  cout << std::endl;

  stbi_write_png("image_grayscale.png", width, height, 1, u8_image_out.data(), width);
  
  // Reclaim now unused memory
  stbi_image_free(u8_image_in);
  //#endif
  //sycl::free(u8_out, sycl_que);
  cout << "Successfully completed on device.\n";
  return 0;
}
