#include <stdio.h>
#include "trt_infer.h"

#include <opencv2/opencv.hpp>
#include <vector>

#include <algorithm>
#include <functional>

std::vector<float> RGB_HWC2CHW(const cv::Mat& hwc_img) {
    int H = hwc_img.rows;
    int W = hwc_img.cols;
    int spatial_size = H * W;

    std::vector<float> chw_buffer(3 * spatial_size);
    
    float* r_ptr = chw_buffer.data();
    float* g_ptr = r_ptr + spatial_size;
    float* b_ptr = g_ptr + spatial_size;

    const float r_mean = 0.485f, g_mean = 0.456f, b_mean = 0.406f;
    const float r_inv_std = 1.0f / 0.229f, g_inv_std = 1.0f / 0.224f, b_inv_std = 1.0f / 0.225f;
    const float scale = 1.0f / 255.0f;

    const uint8_t* src = hwc_img.ptr<uint8_t>();

    for (int i = 0; i < spatial_size; ++i) {
        float r_val = static_cast<float>(src[0]) * scale;
        float g_val = static_cast<float>(src[1]) * scale;
        float b_val = static_cast<float>(src[2]) * scale;

        // The multiply is faster than divide
        *r_ptr++ = (r_val - r_mean) * r_inv_std;
        *g_ptr++ = (g_val - g_mean) * g_inv_std;
        *b_ptr++ = (b_val - b_mean) * b_inv_std;

        src += 3;
    }

    return chw_buffer;
}

int main(int argc, char* argv[]) {
  TrtInfer* classify_model = new TrtInfer();
  classify_model->Build("/home/lamlt/Documents/research/tensorrt/model.engine");
  classify_model->GetEngineInfo();

  cv::Mat input_image = cv::imread("/home/lamlt/Downloads/golden-retriever-tongue-out.jpg", cv::IMREAD_COLOR);
  if (input_image.empty())
    return -1;

  cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
  cv::resize(input_image, input_image, cv::Size(128, 128),0,0, cv::INTER_AREA);
  printf("Pixel (uchar) - R: %d, G: %d, B: %d\n", (int)input_image.data[0], (int)input_image.data[1], (int)input_image.data[2]);
  printf("Pixel (uchar) - R: %d, G: %d, B: %d\n", (int)input_image.data[3], (int)input_image.data[4], (int)input_image.data[5]);
  printf("Pixel (uchar) - R: %d, G: %d, B: %d\n", (int)input_image.data[3*128*128-3], (int)input_image.data[3*128*128-2], (int)input_image.data[3*128*128-1]);

  //printf("%d %d %d\n", )
  
  std::vector<float> transformed_data_image = RGB_HWC2CHW(input_image);
  std::ifstream file("/home/lamlt/Documents/research/tensorrt/my_array.bin", std::ios::binary | std::ios::ate);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<float> transformed_data(fileSize/sizeof(float));

  if (file.read(reinterpret_cast<char*>(transformed_data.data()), fileSize)) {
    printf("read file done %ld\n", transformed_data.size());
  } else {
    printf("faile\n");
  }

  std::vector<float> result(transformed_data.size());
  printf("transformed_data_image size %ld\n", transformed_data_image.size());

  std::transform(transformed_data_image.begin(), transformed_data_image.end(), transformed_data.begin(), result.begin(), std::minus<float>());

  for (int i = 0; i <128; i++) {
    printf("%f ",  result[i]);
  }

  printf("\n ");
  


  // printf("transformed_data size %ld \n", transformed_data.size());
  std::vector<InputData> list_input_data = {{"input", transformed_data_image.data(), 1} };
  classify_model->Infer(list_input_data);

  std::vector<float>out;
  classify_model->GetInferOutput("output", out);

  for (int i = 0; i < out.size(); i++) {
    printf("%f\n", out[i]);
  }


  // printf("%f %f %f ...  %f %f %f\n",  transformed_data[0], transformed_data[1], transformed_data[2],
  //                                     transformed_data[125], transformed_data[126], transformed_data[127]);
  // printf("%f %f %f ...  %f %f %f\n",  transformed_data[0 + 128], transformed_data[1 + 128], transformed_data[2 + 128],
  //                                     transformed_data[125 + 128], transformed_data[126 + 128], transformed_data[127 + 128]);


  return 0;
}