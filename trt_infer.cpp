#include "trt_infer.h"

static Logger gLogger;

TrtInfer::TrtInfer() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

TrtInfer::~TrtInfer() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

int TrtInfer::ReadEngineFile(const std::string& filename, std::vector<char>& buffer)  {
  buffer.clear();
  std::ifstream file(filename, std::ios::binary | std::ios::ate);

  if (!file) {
    return -1;
  }

  size_t size = file.tellg();
  file.seekg(0);

  buffer.resize(size);
  file.read(buffer.data(), size);

  return buffer.size();
}

size_t TrtInfer::GetDimSizeToByte(const nvinfer1::Dims& dims) {
  size_t v = 1;

  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return -1;
    }

    v *= dims.d[i];
  }

  return v;
}

void TrtInfer::AllocateLayerBuffer() {
  printf("======= Start to allocate layer buffer ======\n");
  nvinfer1::ICudaEngine* engine = this->engine_.get();
  int nb = engine->getNbIOTensors();

  for (int i = 0; i < nb; ++i) {
    
    std::string layer_name = std::string(engine->getIOTensorName(i));
    layer_buffer_names_[layer_name] = TensorBuffer();
    layer_buffer_names_[layer_name].dims  = engine->getTensorShape(layer_name.c_str());
    layer_buffer_names_[layer_name].dtype = engine->getTensorDataType(layer_name.c_str());
    layer_buffer_names_[layer_name].mode  = engine->getTensorIOMode(layer_name.c_str());

    size_t element_size = 0;

    switch (layer_buffer_names_[layer_name].dtype) {
      case nvinfer1::DataType::kFLOAT:
        element_size = sizeof(float);
        break;

      case nvinfer1::DataType::kHALF:
        element_size = sizeof(uint16_t);
        break;

      case nvinfer1::DataType::kINT32:
        element_size = sizeof(int32_t);
        break;

      case nvinfer1::DataType::kINT64:
        element_size = sizeof(int64_t);
        break;

      case nvinfer1::DataType::kINT8:
        element_size = sizeof(int8_t);
        break;

      case nvinfer1::DataType::kBOOL:
        element_size = sizeof(bool);
        break;

      default:
          break;
    }

    size_t bytes = 1;
    bool is_dynamic_batch = false;
    for (int i = 0; i < layer_buffer_names_[layer_name].dims.nbDims; ++i) {

      if (layer_buffer_names_[layer_name].dims.d[i] < 0) {
        is_dynamic_batch = true;
      }

      bytes *= layer_buffer_names_[layer_name].dims.d[i];
    }

    bytes *= element_size;
    layer_buffer_names_[layer_name].bytes = abs(bytes);

    if (is_dynamic_batch) {
      layer_buffer_names_[layer_name].bytes = abs(bytes) * max_batch_size_;

      if (layer_buffer_names_[layer_name].mode == nvinfer1::TensorIOMode::kINPUT) {
        nvinfer1::Dims initial_input_dim = layer_buffer_names_[layer_name].dims;
        initial_input_dim.d[0] = current_batch_size_;
        if (!context_->setInputShape(layer_name.c_str(), initial_input_dim)) {
          continue;
        }
      }
    }

    if (layer_buffer_names_[layer_name].bytes > 0) {
      printf("allocate layer %s : %ld bytes\n", layer_name.c_str(), layer_buffer_names_[layer_name].bytes);
      CUDA_CHECK(
        cudaMalloc(
            (void**)&layer_buffer_names_[layer_name].data,
            layer_buffer_names_[layer_name].bytes)
      );

      if (!this->context_->setTensorAddress(layer_name.c_str(), layer_buffer_names_[layer_name].data)) {
        continue;
      }
    }
  }
}

void TrtInfer::GetEngineInfo() {
  printf("\n=== TensorRT Engine ===\n");

  for (const auto& [layer_name, buffer_info] : this->layer_buffer_names_) {

    std::string shape = "[";
    for (int j = 0; j < buffer_info.dims.nbDims; ++j) {
      shape += std::to_string(buffer_info.dims.d[j]) + ", ";
    }
    shape += " ]";

    printf("layer mode: %s || layer name: %s || shape: %s\n",
            buffer_info.mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " : "OUTPUT",
            layer_name.c_str(),
            shape.c_str());
  }
}

int TrtInfer::Build(const std::string& engine_path) {
  if (engine_path == "") {
    gLogger.log(nvinfer1::ILogger::Severity::kWARNING, "engine_path could not be NULL");
    return -1;
  }

  std::vector<char> _engine_data;
  if (this->ReadEngineFile(engine_path, _engine_data) <= 0) {
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Cannot parser the engine data from file");
    return -1;
  }

  this->runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  if (!this->runtime_) {
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create TensorRT runtime");
  }

  nvinfer1::ICudaEngine* _raw_engine = this->runtime_->deserializeCudaEngine(
                                                  _engine_data.data(),
                                                  _engine_data.size()
                                                );
  if (!_raw_engine) {
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to deserialize engine");
    return -1;
  }

  this->engine_.reset(_raw_engine);
  this->context_.reset(this->engine_->createExecutionContext());

  if (!this->context_) {
    gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to create execution context");
    return -1;
  }

  this->AllocateLayerBuffer();
  return 0;
}

int TrtInfer::Infer(const std::vector<InputData>& _input_data) {
  if (_input_data.size() == 0) {
    printf("There is no data pushed to Infer.");
    return -1;
  }

  int check_batch = _input_data[0].batch;

  for (const InputData& data : _input_data) {
    if (data.batch != check_batch) {
      printf("Invalid batch in layer %s with batch %d, the valid batch is %d \n.", data.layer_name.c_str(), data.batch, check_batch);
      return -1;
    }
  }

  if (check_batch != current_batch_size_) {
    for (const InputData& data : _input_data) {
      if (layer_buffer_names_[data.layer_name].mode == nvinfer1::TensorIOMode::kINPUT) {
        nvinfer1::Dims initial_input_dim = layer_buffer_names_[data.layer_name].dims;
        initial_input_dim.d[0] = check_batch;
        if (!context_->setInputShape(data.layer_name.c_str(), initial_input_dim)) {
          return -1;
        }
      } else {
        return -1;
      }
    }

    current_batch_size_ = check_batch;
  }
  for (const InputData& data : _input_data) {

    printf("%f %f %f\n",data.input_data[0],data.input_data[1],data.input_data[2]);
    printf("%f %f %f\n",data.input_data[3*128*128 -3],data.input_data[3*128*128 -2],data.input_data[3*128*128 -1]);
    CUDA_CHECK(cudaMemcpyAsync(
                  layer_buffer_names_[data.layer_name].data,
                  (const void *)data.input_data,
                  layer_buffer_names_[data.layer_name].bytes * current_batch_size_ / max_batch_size_,
                  cudaMemcpyHostToDevice,
                  stream_));
  }

  if (!context_->enqueueV3(stream_)) {
    return -1;
  }

  return 0;
}

size_t TrtInfer::GetInferOutput(const std::string& _layer_name, std::vector<float>& _output) {
  _output.clear();
  size_t output_size = layer_buffer_names_[_layer_name].bytes * current_batch_size_ / max_batch_size_ / sizeof(float);
  printf("output_size %ld\n", output_size);
  _output.resize(output_size);

  CUDA_CHECK(cudaMemcpyAsync(
              _output.data(),
              layer_buffer_names_[_layer_name].data,
              layer_buffer_names_[_layer_name].bytes * current_batch_size_ / max_batch_size_,
              cudaMemcpyDeviceToHost,
              stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  return output_size;
}