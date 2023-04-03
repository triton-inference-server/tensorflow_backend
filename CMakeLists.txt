# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritontensorflowbackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend." ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend." ON)
set(TRITON_TENSORFLOW_VERSION "2" CACHE STRING "TensorFlow version, must be '2'. Starting from 23.04, Triton no longer supports Tensorflow 1.")
set(TRITON_TENSORFLOW_DOCKER_IMAGE "" CACHE STRING "Docker image containing the TensorFlow build required by backend.")
option(TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS "Install extra dependencies directly into the TensorFlow backend, instead of assuming they are present on the system." OFF)
set(TRITON_TENSORFLOW_LIB_PATHS "" CACHE PATH "Paths to TensorFlow libraries. Multiple paths may be specified by separating them with a semicolon.")
set(TRITON_TENSORFLOW_INCLUDE_PATHS "" CACHE PATH "Paths to TensorFlow includes. Multiple paths may be specified by separating them with a semicolon.")

set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo.")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo.")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_TENSORFLOW_DOCKER_BUILD OFF)
if(TRITON_TENSORFLOW_LIB_PATHS STREQUAL "")
  if (TRITON_TENSORFLOW_DOCKER_IMAGE STREQUAL "")
    message(FATAL_ERROR "Using the Tensorflow docker based build requires TRITON_TENSORFLOW_DOCKER_IMAGE")
  endif()
  set(TRITON_TENSORFLOW_DOCKER_BUILD ON)
endif()
message(STATUS "Using Tensorflow docker: ${TRITON_TENSORFLOW_DOCKER_BUILD}")

set(TRITON_TENSORFLOW_CC_LIBNAME "libtensorflow_cc.so")
set(TRITON_TENSORFLOW_FW_LIBNAME "libtensorflow_framework.so")
set(TRITON_TENSORFLOW_LDFLAGS "-ltensorflow_cc -ltensorflow_framework")
if(${TRITON_TENSORFLOW_VERSION} EQUAL "1")
  message(FATAL_ERROR "Starting from 23.04, Triton no longer supports TensorFlow 1. Please switch to TensorFlow 2.")
elseif(${TRITON_TENSORFLOW_VERSION} EQUAL "2")
  set(TRITON_TENSORFLOW_BACKEND_LIBNAME triton_tensorflow2)
  set(TRITON_TENSORFLOW_BACKEND_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/backends/tensorflow2)
  set(TRITON_TENSORFLOW_PYTHON_PATH "/usr/local/lib/python3.8/dist-packages/tensorflow")
else()
  message(FATAL_ERROR "TRITON_TENSORFLOW_VERSION allowed values is '2'")
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
# tensorflow_backend_tf.cc is copied and built into the TensorFlow
# container build (which is independent of any of the Triton
# build). This is done to ensure that it uses the same protobuf as
# TensorFlow. It is kept in this repo only as
# reference. tensorflow_backend_tf.h is also copied into TensorFlow
# build and it used here as the interface to TensorFlow.
#
configure_file(src/libtriton_tensorflow.ldscript libtriton_tensorflow.ldscript COPYONLY)

if (${TRITON_TENSORFLOW_DOCKER_BUILD})
  if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(LIBS_ARCH "aarch64")
  else()
    set(LIBS_ARCH "x86_64")
  endif()

  add_custom_command(
    OUTPUT
      ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
      ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
      LICENSE.tensorflow
    COMMAND docker pull ${TRITON_TENSORFLOW_DOCKER_IMAGE}
    COMMAND docker rm tensorflow_backend_tflib || echo "error ignored..." || true
    COMMAND docker create --name tensorflow_backend_tflib ${TRITON_TENSORFLOW_DOCKER_IMAGE}
    COMMAND docker cp -L tensorflow_backend_tflib:/usr/local/lib/tensorflow/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
    COMMAND docker cp tensorflow_backend_tflib:${TRITON_TENSORFLOW_PYTHON_PATH}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
    COMMAND docker cp tensorflow_backend_tflib:/opt/tensorflow/tensorflow-source/LICENSE LICENSE.tensorflow
    COMMAND docker rm tensorflow_backend_tflib

    COMMAND docker stop tensorflow_backend_deps || echo "error ignored..." || true
    COMMAND docker rm tensorflow_backend_deps || echo "error ignored..." || true
    COMMAND if [ "${TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS}" = "ON" ] \; then mkdir tf_backend_deps && docker run -it -d --name tensorflow_backend_deps ${TRITON_TENSORFLOW_DOCKER_IMAGE} \; fi \;
    COMMAND if [ "${TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS}" = "ON" ] \; then docker exec tensorflow_backend_deps sh -c  "tar -cf - /usr/lib/${LIBS_ARCH}-linux-gnu/libnccl.so*" | tar --strip-components=3 -xf - -C ./tf_backend_deps \; fi
    COMMAND if [ "${TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS}" = "ON" ] \; then docker stop tensorflow_backend_deps && docker rm tensorflow_backend_deps \; fi \;

    COMMENT "Extracting ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} and ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} from ${TRITON_TENSORFLOW_DOCKER_IMAGE}"
  )

  add_custom_target(
    tflib_target
    DEPENDS
      ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
      ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
  )

  add_library(tflib_cc SHARED IMPORTED GLOBAL)
  add_dependencies(tflib_cc tflib_target)
  set_target_properties(
    tflib_cc
    PROPERTIES
      IMPORTED_LOCATION ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
  )

  add_library(tflib_fw SHARED IMPORTED GLOBAL)
  add_dependencies(tflib_fw tflib_target)
  set_target_properties(
    tflib_fw
    PROPERTIES
      IMPORTED_LOCATION ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
  )
endif() # TRITON_TENSORFLOW_DOCKER_BUILD

add_library(
  triton-tensorflow-backend SHARED
  src/tensorflow.cc
  src/tensorflow_utils.cc
  src/tensorflow_utils.h
  src/tensorflow_backend_tf.h
)

add_library(
  TritonTensorFlowBackend::triton-tensorflow-backend ALIAS triton-tensorflow-backend
)

target_include_directories(
  triton-tensorflow-backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (NOT ${TRITON_TENSORFLOW_DOCKER_BUILD})
  target_include_directories(
    triton-tensorflow-backend
    PRIVATE ${TRITON_TENSORFLOW_INCLUDE_PATHS}
  )
endif() # !TRITON_TENSORFLOW_DOCKER_BUILD

target_compile_features(triton-tensorflow-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-tensorflow-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

# C/C++ defines that are used directly by this backend.
if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-tensorflow-backend
    PRIVATE TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU

set_target_properties(
  triton-tensorflow-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ${TRITON_TENSORFLOW_BACKEND_LIBNAME}
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorflow.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_tensorflow.ldscript"
)

if (NOT ${TRITON_TENSORFLOW_DOCKER_BUILD})
  FOREACH(p ${TRITON_TENSORFLOW_LIB_PATHS})
    set(TRITON_TENSORFLOW_LDFLAGS ${TRITON_TENSORFLOW_LDFLAGS} "-L${p}")
  ENDFOREACH(p)
endif() # !TRITON_TENSORFLOW_DOCKER_BUILD


target_link_libraries(
  triton-tensorflow-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

if (${TRITON_TENSORFLOW_DOCKER_BUILD})
  target_link_libraries(
    triton-tensorflow-backend
    PRIVATE
      tflib_cc
      tflib_fw
  )
else()
  target_link_libraries(
    triton-tensorflow-backend
    PRIVATE ${TRITON_TENSORFLOW_LDFLAGS}
  )
endif() # TRITON_TENSORFLOW_DOCKER_BUILD

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-tensorflow-backend
    PRIVATE
      CUDA::cudart
  )
endif() # TRITON_ENABLE_GPU

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonTensorFlowBackend)

install(
  TARGETS
    triton-tensorflow-backend
  EXPORT
    triton-tensorflow-backend-targets
  LIBRARY DESTINATION ${TRITON_TENSORFLOW_BACKEND_INSTALLDIR}
  ARCHIVE DESTINATION ${TRITON_TENSORFLOW_BACKEND_INSTALLDIR}
)

if (${TRITON_TENSORFLOW_DOCKER_BUILD})
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
      ${CMAKE_CURRENT_BINARY_DIR}/${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION}
      ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.tensorflow
    DESTINATION ${TRITON_TENSORFLOW_BACKEND_INSTALLDIR}
  )
  install(
    CODE
      "EXECUTE_PROCESS(
        COMMAND ln -sf ${TRITON_TENSORFLOW_CC_LIBNAME}.${TRITON_TENSORFLOW_VERSION} ${TRITON_TENSORFLOW_CC_LIBNAME}
        COMMAND ln -sf ${TRITON_TENSORFLOW_FW_LIBNAME}.${TRITON_TENSORFLOW_VERSION} ${TRITON_TENSORFLOW_FW_LIBNAME}
        RESULT_VARIABLE TTB_INSTALL_STATUS
        WORKING_DIRECTORY ${TRITON_TENSORFLOW_BACKEND_INSTALLDIR})
      if(TTB_INSTALL_STATUS AND NOT TTB_INSTALL_STATUS EQUAL 0)
        message(FATAL_ERROR \"FAILED: to run patchelf and create links\")
      endif()"
  )

  if(${TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS})
    install(
      DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tf_backend_deps/
      DESTINATION ${TRITON_TENSORFLOW_BACKEND_INSTALLDIR}
    )
  endif() # TRITON_TENSORFLOW_INSTALL_EXTRA_DEPS
endif() # TRITON_TENSORFLOW_DOCKER_BUILD

install(
  EXPORT
    triton-tensorflow-backend-targets
  FILE
    TritonTensorFlowBackendTargets.cmake
  NAMESPACE
    TritonTensorFlowBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonTensorFlowBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorFlowBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorFlowBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-tensorflow-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorFlowBackendTargets.cmake
  NAMESPACE TritonTensorFlowBackend::
)

export(PACKAGE TritonTensorFlowBackend)
