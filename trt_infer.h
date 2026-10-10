#ifndef TRT_INFER_H_
#define TRT_INFER_H_

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cstring>

#define CUDA_CHECK(call)                                                   \
  do {                                                                   \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                    \
  } while (0)

template <typename T>
struct TRTDeleter {
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter<T>>;

struct TensorBuffer {
    void* data = nullptr;
    size_t bytes = 0;
    nvinfer1::TensorIOMode mode;
    nvinfer1::Dims dims{};
    nvinfer1::DataType dtype;
};

struct InputData {
  std::string layer_name;
  float* input_data;
  int batch;
};

class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity <= nvinfer1::ILogger ::Severity::kINFO) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }
};

class TrtInfer {
 private:
  int ReadEngineFile(const std::string& filename, std::vector<char>&);
  size_t GetDimSizeToByte(const nvinfer1::Dims& dims);
  void AllocateLayerBuffer();

 public:
  TrtInfer();
  ~TrtInfer();
  int Build(const std::string&);
  void GetEngineInfo();

  int Infer(const std::vector<InputData>&);
  size_t GetInferOutput(const std::string&, std::vector<float>&);

  int GetCurrentBatchSize() const {
    return current_batch_size_;
  }

  int SetCurrentBatchSize(int batch) {
    if (batch < 0 || batch > max_batch_size_) {
      printf("invalid batch size.\n");
      return -1;
    }

    current_batch_size_ = batch;
    return batch;
  }

 private:
  std::unordered_map<std::string, TensorBuffer> layer_buffer_names_;
  int max_batch_size_ = 4;
  int current_batch_size_ = 1;
  cudaStream_t stream_;
  TRTUniquePtr<nvinfer1::IRuntime> runtime_;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine_;
  TRTUniquePtr<nvinfer1::IExecutionContext> context_;
};

#endif //TRT_INFER_H_