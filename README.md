<!--
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# TensorFlow Backend

The Triton backend for TensorFlow.  You can learn more about backends
in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems in the main Triton [issues
page](https://github.com/triton-inference-server/server/issues).

## Frequently Asked Questions

Full documentation is included below but these shortcuts can help you
get started in the right direction.

### Where can I ask general questions about Triton and Triton backends?

Be sure to read all the information below as well as the [general
Triton
documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main
[server](https://github.com/triton-inference-server/server) repo. If
you don't find your answer there you can ask questions on the main
Triton [issues
page](https://github.com/triton-inference-server/server/issues).

### What versions of TensorFlow are supported by this backend?

The TensorFlow backend supports both TensorFlow 1.x and 2.x. Each
release of Triton will container support for a specific 1.x and 2.x
version. You can find the specific version supported for any release
by checking the Release Notes which are available from the main
[server](https://github.com/triton-inference-server/server) repo.

### Is the TensorFlow backend configurable?

Each model's configuration can enabled [TensorFlow-specific
optimizations](https://github.com/triton-inference-server/server/blob/master/docs/optimization.md#framework-specific-optimization).
There are also a few [command-line options](#command-line-options)
that can be used to configure the backend when launching Triton.

### How do I build the TensorFlow backend?

See [build instructions](#build-the-tensorflow-backend) below.

### Can I use any version of TensorFlow when building the backend?

Currently you must use a version of TensorFlow from
[NGC](https://ngc.nvidia.com). See [custom TensorFlow build
instructions](#build-the-tensorflow-backend-with-custom-tensorflow)
below.

### How does the TensorFlow backend manage GPU memory?

The TensorFlow backend does not "release" GPU memory until the Triton process
exits. TensorFlow uses a pool allocator and so it retains any memory it
allocates until its own process exits. It will reuse that memory if you load
another TensorFlow model, but it will not return it to the system, even if it
is no longer using it. For this reason, it is preferred to keep TensorFlow
models grouped together on the same Triton process if you will be repeatedly
loading/unloading them.

From the TensorFlow GPU docs: "[Memory is not released since it can lead to memory fragmentation](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)".

#### Workarounds

The following are a few available options to limit the total amount of memory
that TensorFlow allocates:

1. You can use `gpu-memory-fraction` as described 
[here](https://github.com/triton-inference-server/tensorflow_backend#--backend-configtensorflowgpu-memory-fractionfloat).
This restricts an upper-bound on the total memory TensorFlow can allocate for
the process. However, note when using this option that allow-growth is set to
false, hence running TF models might still fail if TF needs
to allocate more memory for its executions than what's allowed. 

2. To limit large growths in memory from concurrent TensorFlow executions,
you can also use the [rate limiter](https://github.com/triton-inference-server/server/blob/main/docs/rate_limiter.md)
in Triton to limit the number of requests allowed to enter execution.

## Auto-Complete Model Configuration

Assuming Triton was not started with `--disable-auto-complete-config` command line
option, the Tensorflow Backend makes use of the metadata available in TensorFlow
SavedModel to populate the required fields in the model's config.pbtxt. You can
learn more about Triton's support for auto-completing model configuration from
[here](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration).

However, in Graphdef format, models do not carry sufficient metadata and hence
Triton cannot generate model configuration for them. As a result, config.pbtxt
must be provided for such models explicitly.

Tensorflow backend can complete the following fields in model configuration:

### max_batch_size

Auto-completing max_batch_size follows the following rules:

1. Autocomplete has determined the model is capable of batching requests.
2. max_batch_size is 0 in the model configuration or max_batch_size
   is omitted from the model configuration.

If the above two rules are met, max_batch_size is set to
[default-max-batch-size](#--backend-config=tensorflow,default-max-batch-size=\<int\>).
Otherwise max_batch_size is set as 0.


### Inputs and Outputs

The Tensorflow Backend is able to fill in the `name`, `data_type`, and `dims` provided this
information is available in the model. Known limitations are inputs which are defined in
the [`ragged_batching`](https://github.com/triton-inference-server/server/blob/main/docs/ragged_batching.md#batch-input) and
[`sequence_batching`](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#sequence-batcher)
fields. There is not enough information in the model for the backend to be able to autocomplete these.
Additionally, the backend cannot auto complete configuration for scalar tensors.

Autocompleting outputs follows the following rules:
- If `outputs` is empty or undefined in the model configuration, all outputs in the savedmodel 
will be autocompleted
- If one or more output is defined in `outputs`, those outputs which are defined will be
autocompleted and those which are omitted will be ignored.

### Dynamic Batching

If max_batch_size > 1 and no [scheduler](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#scheduling-and-batching)
is provided, the dynamic batch scheduler will be enabled with default settings.

## Command-line Options

The command-line options configure properties of the TensorFlow
backend that are then applied to all models that use the backend.

##### --backend-config=tensorflow,allow-soft-placement=\<boolean\>

Instruct TensorFlow to use CPU implementation of an operation when a
GPU implementation is not available.

##### --backend-config=tensorflow,gpu-memory-fraction=\<float\>

Reserve a portion of GPU memory for TensorFlow models. Default value
0.0 indicates that TensorFlow should dynamically allocate memory as
needed. Value of 1.0 indicates that TensorFlow should allocate all of
GPU memory.

##### --backend-config=tensorflow,version=\<int\>

Select the version of the TensorFlow library to be used, available
versions are 1 and 2. Default version is 2.

##### --backend-config=tensorflow,default-max-batch-size=\<int\>

The default value to use for max_batch_size during [auto-completing model configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration)
when batching support is detected in the model. Note that if not
explicitly provided, the default value for this option is 4.

## Build the TensorFlow Backend

Use a recent cmake to build. First install the required dependencies.

```
$ apt-get install patchelf rapidjson-dev
```

The backend can be built to support either TensorFlow 1.x or
TensorFlow 2.x. An appropriate TensorFlow container from
[NGC](https://ngc.nvidia.com) must be used. For example, to build a backend
that uses the 21.02 version of the TensorFlow 1.x container from NGC:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_TENSORFLOW_VERSION=1 -DTRITON_TENSORFLOW_DOCKER_IMAGE="nvcr.io/nvidia/tensorflow:21.02-tf1-py3" ..
$ make install
```

For example, to build a backend that uses the 21.02 version of the
TensorFlow 2.x container from NGC:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_TENSORFLOW_VERSION=2 -DTRITON_TENSORFLOW_DOCKER_IMAGE="nvcr.io/nvidia/tensorflow:21.02-tf2-py3" ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Build the TensorFlow Backend With Custom TensorFlow

Currently, Triton requires that a specially patched version of
TensorFlow be used with the TensorFlow backend. The full source for
these TensorFlow versions are available as Docker images from
[NGC](https://ngc.nvidia.com). For example, the TensorFlow 1.x version
compatible with the 21.02 release of Triton is available as
nvcr.io/nvidia/tensorflow:21.02-tf1-py3 and the TensorFlow 2.x version
compatible with the 21.02 release of Triton is available as
nvcr.io/nvidia/tensorflow:21.02-tf2-py3.

You can modify and rebuild TensorFlow within these images to generate
the shared libraries needed by the Triton TensorFlow backend. In the
TensorFlow 1.x or TensorFlow 2.x container you rebuild using:

```
$ /opt/tensorflow/nvbuild.sh
```

After rebuilding within the container you should save the updated
container as a new Docker image (for example, by using *docker
commit*), and then build the backend as described
[above](#build-the-tensorflow-backend) with
TRITON_TENSORFLOW_DOCKER_IMAGE set to refer to the new Docker image.


## Using the Tensorflow Backend
### Parameters

Configuration of Tensorflow for a model is done through the Parameters section of the model's 'config.pbtxt' file. The parameters and their description are as follows.

* `TF_NUM_INTRA_THREADS`: Number of threads to use to parallelize the execution of an individual op. Auto-configured by default. See [protobuf here](https://github.com/tensorflow/tensorflow/blob/6f72753a66d6abab8b839cc263a9f1329861f6f9/tensorflow/core/protobuf/config.proto#L393). Should be a non-negative number.
* `TF_NUM_INTER_THREADS`: Controls the number of operators that can be executed simultaneously. Auto-configured by default. See [protobuf here](https://github.com/tensorflow/tensorflow/blob/6f72753a66d6abab8b839cc263a9f1329861f6f9/tensorflow/core/protobuf/config.proto#L404). Should be a non-negative number.
* `TF_USE_PER_SESSION_THREADS`: Boolean value to see if per session thread is used. "True", "On" and "1" are accepted as true.
* `TF_GRAPH_TAG`: Tag of the graphs to use. See [protobuf here](https://github.com/tensorflow/tensorflow/blob/6f72753a66d6abab8b839cc263a9f1329861f6f9/tensorflow/core/protobuf/meta_graph.proto#L56)
* `TF_SIGNATURE_DEF`: Signature def to use. See [protobuf here](https://github.com/tensorflow/tensorflow/blob/6f72753a66d6abab8b839cc263a9f1329861f6f9/tensorflow/core/protobuf/meta_graph.proto#L260-L331)
* `MAX_SESSION_SHARE_COUNT`: This parameter specifies the maximum number of [model instances](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#instance-groups) that can share a [TF session](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/session.h). The default value is 1 which means Triton will create a separate TF session for each model instance. If this parameter is set to the total number of instances, then Triton will create only a single TF session which will be shared by all the instances. Sharing TF sessions among model instances can reduce memory footprint of loading and executing the model.
* `TF_INIT_OPS_FILE`: This parameter specifies the name of the file in JSON
format that contains the [initialization operations](https://www.tensorflow.org/api_docs/python/tf/compat/v1/global_variables_initializer).
The JSON file must have a single element named 'init_ops' which describes the
list of initialization operations. This file can be stored in the model version
folder or in the model directory. If it is provided in both locations, the model
version folder takes precedence over the one provided in the model folder. If it
is provided in the model version folder, the directory structure should look
like below:

```
|-- 1
|   |-- model.graphdef
|   `-- init_ops.json
`-- config.pbtxt
```

Below is an example of the contents of the `init_ops.json` file.

```json
{
    "init_ops": ["init"]
}
```


The section of model config file specifying these parameters will look like:

```
parameters: {
  key: "TF_NUM_INTRA_THREADS"
  value: {
    string_value:"2"
  }
}

parameters: {
  key: "TF_USE_PER_SESSION_THREADS"
  value: {
    string_value:"yes"
  }
}

parameters: {
  key: "TF_GRAPH_TAG"
  value: {
    string_value: "serve1"
  }
}

parameters: {
  key: "TF_INIT_OPS_FILE"
  value: {
    string_value: "init_ops.json"
  }
}

parameters: {
  key: "TF_SIGNATURE_DEF"
  value: {
    string_value: "serving2"
  }
}
```
