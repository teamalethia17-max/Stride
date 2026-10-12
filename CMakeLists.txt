cmake_minimum_required(VERSION 3.20)
project(stride VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(STRIDE_ENABLE_OPENMP "Enable OpenMP for matrix kernels when available" ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /EHsc /Zc:__cplusplus)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(stride_core
  src/tensor.cpp
  src/ops.cpp
  src/test_model.cpp
)
target_include_directories(stride_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(stride_core PUBLIC
  STRIDE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  STRIDE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  STRIDE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  STRIDE_VERSION_STRING="${PROJECT_VERSION}"
)

if(STRIDE_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(stride_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(stride_core PUBLIC STRIDE_HAS_OPENMP=1)
  endif()
endif()

add_executable(stride src/main.cpp)
target_link_libraries(stride PRIVATE stride_core)

enable_testing()

function(stride_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE stride_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

stride_add_test(test_tensor)
stride_add_test(test_ops)

add_test(NAME cli_help COMMAND stride --help)
add_test(NAME cli_version COMMAND stride --version)
add_test(NAME cli_test_model COMMAND stride test-model)

install(TARGETS stride RUNTIME DESTINATION bin)
