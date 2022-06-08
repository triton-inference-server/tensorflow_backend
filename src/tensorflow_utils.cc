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

#include "tensorflow_utils.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace tensorflow {

bool
ModelSupportsBatch(std::vector<const TRITONTF_IOList*> model_ios)
{
  for (const auto& ios : model_ios) {
    for (const TRITONTF_IOList* itr = ios; itr != nullptr; itr = itr->next_) {
      TRITONTF_IO* io = itr->io_;
      if ((io->shape_->rank_) != 0 && (io->shape_->dims_[0] != -1)) {
        return false;
      }
    }
  }

  return true;
}

TRITONSERVER_Error*
CompareDims(
    const std::string& model_name, const std::string& tensor_name,
    const TRITONTF_Shape* model_shape, const std::vector<int64_t>& dims,
    const bool supports_batching, const bool compare_exact)
{
  // If the model configuration expects batching support in the model,
  // then the tensorflow shape first dimension must be -1.
  if (supports_batching) {
    if ((model_shape->rank_ == 0) || (model_shape->dims_[0] != WILDCARD_DIM)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': for the model to support batching the shape should have at "
           "least 1 dimension and the first dimension must be -1; but shape "
           "expected by the model is " +
           ShapeToString(model_shape))
              .c_str());
    }

    std::vector<int64_t> full_dims;
    full_dims.emplace_back(WILDCARD_DIM);
    full_dims.insert(full_dims.end(), dims.begin(), dims.end());

    bool succ = (model_shape->rank_ == (size_t)full_dims.size());
    if (succ) {
      for (size_t i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape->dims_[i];
        if (compare_exact || (model_dim != WILDCARD_DIM)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': the model expects " + std::to_string(model_shape->rank_) +
           " dimensions (shape " + ShapeToString(model_shape) +
           ") but the model configuration specifies " +
           std::to_string(full_dims.size()) +
           " dimensions (an initial batch dimension because max_batch_size "
           "> 0 followed by the explicit tensor shape, making complete "
           "shape " +
           backend::ShapeToString(full_dims) + ")")
              .c_str());
    }
  } else {
    // ! supports_batching
    bool succ = (model_shape->rank_ == (size_t)dims.size());
    if (succ) {
      for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape->dims_[i];
        if (compare_exact || (model_dim != WILDCARD_DIM)) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': the model expects " + std::to_string(model_shape->rank_) +
           " dimensions (shape " + ShapeToString(model_shape) +
           ") but the model configuration specifies " +
           std::to_string(dims.size()) + " dimensions (shape " +
           backend::ShapeToString(dims) + ")")
              .c_str());
    }
  }

  return nullptr;  // success
}

const TRITONTF_IO*
FindIOByName(const TRITONTF_IOList* ios, const std::string& name)
{
  for (const TRITONTF_IOList* itr = ios; itr != nullptr; itr = itr->next_) {
    if (itr->io_->name_ == name) {
      return itr->io_;
    }
  }

  return nullptr;
}

const TRITONTF_IO*
FindIOByName(const std::vector<const TRITONTF_IOList*> ios, std::string& name)
{
  for (const auto itr : ios) {
    if (itr->io_->name_ == name) {
      return itr->io_;
    }
  }

  return nullptr;
}

std::string
ShapeToString(const TRITONTF_Shape* shape, const size_t start_idx)
{
  std::string str("[");
  for (size_t idx = start_idx; idx < shape->rank_; idx++) {
    if (idx > start_idx) {
      str += ",";
    }

    str += std::to_string(shape->dims_[idx]);
  }

  str += "]";
  return str;
}

bool
CompareDataType(TRITONTF_DataType model_dtype, const std::string& dtype)
{
  auto cdtype = ConvertDataType(dtype);
  if (cdtype == TRITONTF_TYPE_INVALID) {
    return false;
  }

  return model_dtype == cdtype;
}

bool
DataTypeIsInvalid(const std::string& dtype)
{
  auto cdtype = ConvertDataType(dtype);
  return cdtype == TRITONTF_TYPE_INVALID;
}

TRITONSERVER_DataType
ConvertDataType(TRITONTF_DataType dtype)
{
  switch (dtype) {
    case TRITONTF_DataType::TRITONTF_TYPE_INVALID:
      return TRITONSERVER_TYPE_INVALID;
    case TRITONTF_DataType::TRITONTF_TYPE_BOOL:
      return TRITONSERVER_TYPE_BOOL;
    case TRITONTF_DataType::TRITONTF_TYPE_UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case TRITONTF_DataType::TRITONTF_TYPE_UINT16:
      return TRITONSERVER_TYPE_UINT16;
    case TRITONTF_DataType::TRITONTF_TYPE_UINT32:
      return TRITONSERVER_TYPE_UINT32;
    case TRITONTF_DataType::TRITONTF_TYPE_UINT64:
      return TRITONSERVER_TYPE_UINT64;
    case TRITONTF_DataType::TRITONTF_TYPE_INT8:
      return TRITONSERVER_TYPE_INT8;
    case TRITONTF_DataType::TRITONTF_TYPE_INT16:
      return TRITONSERVER_TYPE_INT16;
    case TRITONTF_DataType::TRITONTF_TYPE_INT32:
      return TRITONSERVER_TYPE_INT32;
    case TRITONTF_DataType::TRITONTF_TYPE_INT64:
      return TRITONSERVER_TYPE_INT64;
    case TRITONTF_DataType::TRITONTF_TYPE_FP16:
      return TRITONSERVER_TYPE_FP16;
    case TRITONTF_DataType::TRITONTF_TYPE_FP32:
      return TRITONSERVER_TYPE_FP32;
    case TRITONTF_DataType::TRITONTF_TYPE_FP64:
      return TRITONSERVER_TYPE_FP64;
    case TRITONTF_DataType::TRITONTF_TYPE_STRING:
      return TRITONSERVER_TYPE_BYTES;
    default:
      break;
  }

  return TRITONSERVER_TYPE_INVALID;
}

TRITONTF_DataType
ConvertDataType(const std::string& dtype)
{
  if (dtype == "TYPE_INVALID") {
    return TRITONTF_DataType::TRITONTF_TYPE_INVALID;
  } else if (dtype == "TYPE_BOOL") {
    return TRITONTF_DataType::TRITONTF_TYPE_BOOL;
  } else if (dtype == "TYPE_UINT8") {
    return TRITONTF_DataType::TRITONTF_TYPE_UINT8;
  } else if (dtype == "TYPE_UINT16") {
    return TRITONTF_DataType::TRITONTF_TYPE_UINT16;
  } else if (dtype == "TYPE_UINT32") {
    return TRITONTF_DataType::TRITONTF_TYPE_UINT32;
  } else if (dtype == "TYPE_UINT64") {
    return TRITONTF_DataType::TRITONTF_TYPE_UINT64;
  } else if (dtype == "TYPE_INT8") {
    return TRITONTF_DataType::TRITONTF_TYPE_INT8;
  } else if (dtype == "TYPE_INT16") {
    return TRITONTF_DataType::TRITONTF_TYPE_INT16;
  } else if (dtype == "TYPE_INT32") {
    return TRITONTF_DataType::TRITONTF_TYPE_INT32;
  } else if (dtype == "TYPE_INT64") {
    return TRITONTF_DataType::TRITONTF_TYPE_INT64;
  } else if (dtype == "TYPE_FP16") {
    return TRITONTF_DataType::TRITONTF_TYPE_FP16;
  } else if (dtype == "TYPE_FP32") {
    return TRITONTF_DataType::TRITONTF_TYPE_FP32;
  } else if (dtype == "TYPE_FP64") {
    return TRITONTF_DataType::TRITONTF_TYPE_FP64;
  } else if (dtype == "TYPE_STRING") {
    return TRITONTF_DataType::TRITONTF_TYPE_STRING;
  }
  return TRITONTF_DataType::TRITONTF_TYPE_INVALID;
}

std::string
ConvertToModelConfigString(TRITONTF_DataType dtype)
{
  if (dtype == TRITONTF_DataType::TRITONTF_TYPE_INVALID) {
    return "TYPE_INVALID";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_BOOL) {
    return "TYPE_BOOL";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_UINT8) {
    return "TYPE_UINT8";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_UINT16) {
    return "TYPE_UINT16";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_UINT32) {
    return "TYPE_UINT32";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_UINT64) {
    return "TYPE_UINT64";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_INT8) {
    return "TYPE_INT8";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_INT16) {
    return "TYPE_INT16";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_INT32) {
    return "TYPE_INT32";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_INT64) {
    return "TYPE_INT64";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_FP16) {
    return "TYPE_FP16";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_FP32) {
    return "TYPE_FP32";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_FP64) {
    return "TYPE_FP64";
  } else if (dtype == TRITONTF_DataType::TRITONTF_TYPE_STRING) {
    return "TYPE_STRING";
  }
  return "TYPE_INVALID";
}

TRITONTF_DataType
ConvertDataType(TRITONSERVER_DataType dtype)
{
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return TRITONTF_DataType::TRITONTF_TYPE_INVALID;
    case TRITONSERVER_TYPE_BOOL:
      return TRITONTF_DataType::TRITONTF_TYPE_BOOL;
    case TRITONSERVER_TYPE_UINT8:
      return TRITONTF_DataType::TRITONTF_TYPE_UINT8;
    case TRITONSERVER_TYPE_UINT16:
      return TRITONTF_DataType::TRITONTF_TYPE_UINT16;
    case TRITONSERVER_TYPE_UINT32:
      return TRITONTF_DataType::TRITONTF_TYPE_UINT32;
    case TRITONSERVER_TYPE_UINT64:
      return TRITONTF_DataType::TRITONTF_TYPE_UINT64;
    case TRITONSERVER_TYPE_INT8:
      return TRITONTF_DataType::TRITONTF_TYPE_INT8;
    case TRITONSERVER_TYPE_INT16:
      return TRITONTF_DataType::TRITONTF_TYPE_INT16;
    case TRITONSERVER_TYPE_INT32:
      return TRITONTF_DataType::TRITONTF_TYPE_INT32;
    case TRITONSERVER_TYPE_INT64:
      return TRITONTF_DataType::TRITONTF_TYPE_INT64;
    case TRITONSERVER_TYPE_FP16:
      return TRITONTF_DataType::TRITONTF_TYPE_FP16;
    case TRITONSERVER_TYPE_FP32:
      return TRITONTF_DataType::TRITONTF_TYPE_FP32;
    case TRITONSERVER_TYPE_FP64:
      return TRITONTF_DataType::TRITONTF_TYPE_FP64;
    case TRITONSERVER_TYPE_BYTES:
      return TRITONTF_DataType::TRITONTF_TYPE_STRING;
    default:
      break;
  }

  return TRITONTF_DataType::TRITONTF_TYPE_INVALID;
}

TRITONSERVER_Error*
ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    bool* value)
{
  std::string value_str;
  RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
  RETURN_IF_ERROR(ParseBoolValue(value_str, value));

  return nullptr;
}

TRITONSERVER_Error*
ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    int* value)
{
  std::string value_str;
  RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
  RETURN_IF_ERROR(ParseIntValue(value_str, value));

  return nullptr;
}

TRITONSERVER_Error*
ParseParameter(
    triton::common::TritonJson::Value& params, const std::string& mkey,
    std::string* value)
{
  RETURN_IF_ERROR(GetParameterValue(params, mkey, value));
  return nullptr;
}

}}}  // namespace triton::backend::tensorflow
