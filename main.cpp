#include <stdio.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavcodec/avcodec.h>
}

#include "whisper.h"

std::vector<float> decode_mp3_to_ram(const char* filename) {
    std::vector<float> pcm_output;

    AVFormatContext* format_ctx = avformat_alloc_context();
    avformat_open_input(&format_ctx, filename, NULL, NULL);
    avformat_find_stream_info(format_ctx, NULL);

    const AVCodec* codec = nullptr;
    int audio_stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (audio_stream_idx < 0) {
        avformat_close_input(&format_ctx);
        return {};
    }
   
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        avformat_close_input(&format_ctx);
        return {};
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[audio_stream_idx]->codecpar) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return {};
    }

    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return {};
    }

    SwrContext* swr_ctx = nullptr;
    AVChannelLayout out_ch_layout;
    av_channel_layout_default(&out_ch_layout, 1);

    int ret = swr_alloc_set_opts2(
        &swr_ctx,
        &out_ch_layout,
        AV_SAMPLE_FMT_FLT,
        16000,
        &codec_ctx->ch_layout, 
        codec_ctx->sample_fmt, 
        codec_ctx->sample_rate, 
        0, NULL
    );

    if (ret < 0 || !swr_ctx) {
        av_channel_layout_uninit(&out_ch_layout);
        return {};
    }


    swr_init(swr_ctx);

    AVPacket packet;
    AVFrame* frame = av_frame_alloc();

    while (av_read_frame(format_ctx, &packet) >= 0) {
        if (packet.stream_index == audio_stream_idx) {
            if (avcodec_send_packet(codec_ctx, &packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    float* buffer = nullptr;
                    int out_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
                    av_samples_alloc((uint8_t**)&buffer, NULL, 1, out_samples, AV_SAMPLE_FMT_FLT, 0);

                    int converted = swr_convert(swr_ctx, (uint8_t**)&buffer, out_samples, (const uint8_t**)frame->data, frame->nb_samples);

                    pcm_output.insert(pcm_output.end(), buffer, buffer + converted);
                    av_freep(&buffer);
                }
            }
        }
        av_packet_unref(&packet);
    }

    av_frame_free(&frame);
    swr_free(&swr_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    return pcm_output;
}

int main(int argc, char* argv[]) {

  std::vector<float> pcm_data = decode_mp3_to_ram("/workspace/audio.mp3");

  printf("pcm length %ld \n", pcm_data.size());
  whisper_filters  model_filters;
  printf("model_filters.n_fft _ 1 %d \n", model_filters.n_fft);
  whisper_update_filters(argv[1], model_filters);

  printf("model_filters.n_fft %d \n", model_filters.n_fft);

  whisper_mel mel;
  log_mel_spectrogram(pcm_data.data(), pcm_data.size(), model_filters.n_mel, 8, model_filters, true, mel);
  return 0;
}