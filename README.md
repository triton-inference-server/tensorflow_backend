## 非内置模型加载示例

服务使用NVIDIA Triton inference server构建，支持Tensorflow graphdef, Tensorflow savedmodel, Pytorch, TensorRT, ONNX等多种模型。非内置模型需要按Triton[模型仓库](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)的结构组织好，并且在加载时在`DFS目录`选择模型仓库下面模型名称对应的目录。

``` 
 <model-repository-path>/
    <model-name>/  #选择这个目录
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>  #选择这个目录
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
``` 

### Tensorflow graphdef示例
#### 1、模型文件结构
```
model_name
  ├── 1
  │   └── model.graphdef
  └── config.pbtxt

``` 
#### 2、模型配置文件
```
  platform: "tensorflow_graphdef"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### Tensorflow savedmodel示例
#### 1、模型文件结构
```
model_name
   ├── 1
   │   └── model.savedmodel
   │          └── saved_model.pb
   │          ├── variables
   │                └── variables.data-00000-of-00001
   │                ├── variables.index
   ├── config.pbtxt
``` 
#### 2、模型配置文件
```
  platform: "tensorflow_savedmodel"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### Pytorch示例
#### 1、模型文件结构
```
model_name
  ├── 1
  │   └── model.pt
  └── config.pbtxt

``` 
#### 2、模型配置文件
```
  platform: "pytorch_libtorch"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### TensorRT示例
#### 1、模型文件结构
```
modle_name
  ├── 1
  │   └── model.plan
  └── config.pbtxt
``` 
#### 2、模型配置文件
```
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```

### ONNX示例
#### 1、模型文件结构
```
modle_name
  ├── 1
  │   └── model.onnx
  └── config.pbtxt
``` 
#### 2、模型配置文件
```
  platform: "onnxruntime_onnx"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
```
