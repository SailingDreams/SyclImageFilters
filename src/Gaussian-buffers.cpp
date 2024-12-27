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
#include <chrono>
#include "imageUtilsAgnostic.h"
#include "imageUtilsUsingBuffers.h"
#include "imageUtilsUsingCpp.h"
#include "image.h"

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
  //path = "..\\images\\humingBirds.png";
  path = "..\\images\\Bikesgray.jpg"; // see https://en.wikipedia.org/wiki/Sobel_operator
  path = "..\\images\\HummingBirdAtFeeder.png";
  const int LOAD_IMAGE_AS_IS = 0;

  std::chrono::steady_clock::time_point timeBegin;
  std::chrono::steady_clock::time_point timeEnd;

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
  string imgWrittenBackOutStr = "image_as_read_in_";
  imgWrittenBackOutStr += to_string(channels) + "channels.png";
  stbi_write_png("image_as_read_in_3chan.png", width, height, channels, u8_image_in, width * channels);
  
  //uint8_t* u8_out = reinterpret_cast<uint8_t*>(sycl::malloc_shared(width * height, sycl_que));
  std::vector<uint8_t> u8_image_out(width * height);
  //array<uint8_t, 512*512> u8_image_out;
  //cout << "u8_image_out size " << u8_image_out.size() << std::endl;  

  // Target to hold grayscale image. This constructor indicates that the memory should be allocated by the runtime
  vector<float> fl_grayscale(width * height);

  int numIterations = 100;

  { // Set scope for SYCL buffers

    // Create sycl buffer for the input image
    buffer<uint8_t, 1> u8_image_in_buffer{u8_image_in, width * height * channels};
    // Create sycl buffer for the grayscale image
    buffer<float, 1> fl_grayscale_buffer{fl_grayscale.data(),width * height};

    buffer<float, 1> fl_sobel_img_buffer{width * height};

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
      //cout << "Running on platform: "
      //    << sycl_que.get_device().get_info<info::platform::name>() << "\n";
      cout << "Local Memory Size: " 
          << (float)(sycl_que.get_device().get_info<info::device::local_mem_size>())/1024.0f 
          << " kBytes" << std::endl;
      
      #ifdef USE_SYCL
        cout << "Using SYCL" << std::endl;
        // Convert to gray scale range 0 ... 1.0
        ConvertToGrayscaleBuffer(sycl_que, u8_image_in_buffer, fl_grayscale_buffer, width, height,
                                 channels);

        timeBegin = std::chrono::steady_clock::now();
        for(int i = 0; i < numIterations; i++)
        {
          SobelFilter(sycl_que, fl_grayscale_buffer, fl_sobel_img_buffer,
                    width, height);
        }
        timeEnd = std::chrono::steady_clock::now();

        ConvertToUint8Buffer(sycl_que, fl_sobel_img_buffer, u8_image_out_buffer, width, height);
        //ConvertToUint8Buffer(sycl_que, fl_grayscale_buffer, u8_image_out_buffer, width, height);
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
  
    //this line is here to create a compile error if USE_SYCL is undefined
    cout << "Running C++ version" << std::endl;
    // Convert to gray scale range 0 ... 1.0
    ConvertToGrayscaleCpp(u8_image_in, fl_grayscale, width, height, channels);

    std::vector<float> imageMag(width * height);

    timeBegin = std::chrono::steady_clock::now();
    for(int i = 0; i < numIterations; i++)
    {
      SobelFilterCpp(fl_grayscale, imageMag, width, height);
    }
    timeEnd = std::chrono::steady_clock::now();
    ConvertToUint8Cpp(imageMag, u8_image_out, width, height); 
    //ConvertToUint8Cpp(fl_grayscale, u8_image_out, width, height);
    cout << "Done running C++" << std::endl;
  #endif
  
//auto procTimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeBegin);
auto procTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin);
std::cout << "Processing Time " << (float)(procTimeUs.count())/(1000.0f * float(numIterations)) 
                                << " msec]" << std::endl;

#if 0
  // Print out some image values
  cout << "Greyscale image in float" << std::endl;
  for(int i = 0; i < 15; i++) { cout << fl_grayscale[i] << ", ";}
  cout << std::endl;
  cout << "Greyscale image out uint8" << std::endl;
  for(int i = 0; i < 15; i++) { cout << (int)u8_image_out[i] << ", ";}
  cout << std::endl;

  cout << "sobelXOut float" << std::endl;
  for(int i = 0; i < 15; i++) { cout << sobelXOut[i] << ", ";}
  cout << std::endl;    
  cout << "u8_imageX_out uint8" << std::endl;
  for(int i = 0; i < 15; i++) { cout << (int)u8_imageX_out[i] << ", ";}
  cout << std::endl;

  cout << "sobelYOut float" << std::endl;
  for(int i = 0; i < 15; i++) { cout << sobelYOut[i] << ", ";}
  cout << std::endl;  
  cout << "u8_imageY_out uint8" << std::endl;
  for(int i = 0; i < 15; i++) { cout << (int)u8_imageY_out[i] << ", ";}
  cout << std::endl;
#endif

  stbi_write_png("image_grayscale.png", width, height, 1, u8_image_out.data(), width);
  
  // Reclaim now unused memory
  stbi_image_free(u8_image_in);
  //#endif
  //sycl::free(u8_out, sycl_que);
  cout << "Successfully completed on device.\n";
  return 0;
}
