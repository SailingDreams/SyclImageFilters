# Select between the USM and buffer variant of the
# code to compile, depending on the value of USM
# given to cmake
# e.g. if cmake is called with -DUSM=1, the USM
# source code will be compiled
if(DEFINED USM AND (NOT(USM EQUAL 0)))
    message(STATUS "Using the USM variant.")
    set(SOURCE_FILE gaussian-usm.cpp)
    set(TARGET_NAME gaussian-usm)
else()
    set(SOURCE_FILE imageUtilsAgnostic.cpp  
                    imageUtilsUsingCpp.cpp 
                    imageUtilsUsingBuffers.cpp 
                    gaussian-buffers.cpp )
    set(TARGET_NAME gaussian-buffers)
endif()


message(STATUS "CMAKE_SOURCE_DIR  = ${CMAKE_SOURCE_DIR}")
message(STATUS "src directory CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")

# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()



#
# SECTION 1
# This section defines rules to create a cpu-gpu make target
# This can safely be removed if your project is only targetting FPGAs
#

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
#set(LINK_FLAGS "-fsycl")
set(LINK_FLAGS "-fsycl -Wl,--stack,10000000")

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

#
# End of SECTION 1
#

#

