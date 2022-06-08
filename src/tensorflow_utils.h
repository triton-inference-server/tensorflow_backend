// Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "tensorflow_backend_tf.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace tensorflow {

/// \return True if the provided model I/Os allow batching support; False
/// otherwise.
bool ModelSupportsBatch(std::vector<const TRITONTF_IOList*> model_ios);

/// \return nullptr if a TensorFlow shape can support a model
/// configuration shape. Dimensions with variable size in the
/// TensorFlow shape can support any size in the corresponding model
/// configuration shape dimension. Dimensions with variable size in
/// the model configuration shape must be variable size in the
/// TensorFlow shape. All fixed-sized dimensions must match exactly.
/// \param supports_batching If True then the configuration expects
/// the model to support batching and so the shape must have the
/// appropriate batch dimension.
TRITONSERVER_Error* CompareDims(
    const std::string& model_name, const std::string& tensor_name,
    const TRITONTF_Shape* model_shape, const std::vector<int64_t>& dims,
    const bool supports_batching, const bool compare_exact);

/// \return a named input/output tensor. Return nullptr if not found.
const TRITONTF_IO* FindIOByName(
    const TRITONTF_IOList* ios, const std::string& name);

/// \return a named input/output tensor. Return nullptr if not found.
const TRITONTF_IO* FindIOByName(
    const std::vector<const TRITONTF_IOList*> ios, std::string& name);

// Convert a vector representing a shape to string representation.
/// \param dims The vector of dimensions to be converted.
/// \return String representation of the vector in pattern
/// "[d0,d1,...,dn]"
std::string ShapeToString(
    const TRITONTF_Shape* model_shape, const size_t start_idx = 0);

/// \return true if a TF data-type matches a model configuration
/// data-type.
bool CompareDataType(TRITONTF_DataType model_dtype, const std::string& dtype);

/// \return true if a model configuration data-type is invalid
bool DataTypeIsInvalid(const std::string& dtype);

/// \return the TRITONSERVER data-type that corresponds to a
/// TRITONTF data-type.
TRITONSERVER_DataType ConvertDataType(TRITONTF_DataType dtype);

/// \return the model configuration data-type corresponding to a TRITONTF
/// data-type.
std::string ConvertToModelConfigString(TRITONTF_DataType dtype);

/// \return the TRITONTF data-type corresponding to a model
/// configuration data-type.
TRITONTF_DataType ConvertDataType(const std::string& dtype);

/// \return the TRITONTF data-type corresponding to a model
/// configuration data-type.
TRITONTF_DataType ConvertDataType(TRITONSERVER_DataType dtype);

TRITONSERVER_Error* ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    bool* value);
TRITONSERVER_Error* ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    int* value);
TRITONSERVER_Error* ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    std::string* value);

// If TRITONTF Error is non-OK, return the equivalent TRTIS status.
#define RETURN_IF_TRITONTF_ERROR(TFWS)                                       \
  do {                                                                       \
    TRITONTF_Error* error__ = (TFWS);                                        \
    if (error__ != nullptr) {                                                \
      auto status =                                                          \
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error__->msg_); \
      TRITONTF_ErrorDelete(error__);                                         \
      return status;                                                         \
    }                                                                        \
  } while (false)

}}}  // namespace triton::backend::tensorflow
