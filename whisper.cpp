#include "whisper.h"

#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <cassert>
#include <algorithm>

#define GGML_FILE_MAGIC   0x67676d6c

whisper_global_cache global_cache;

static void dft(const float* in, int N, float* out) {
    const int sin_cos_step = SIN_COS_N_COUNT / N;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT); // t = 2*M_PI*k*n/N
            re += in[n]*global_cache.cos_vals[idx]; // cos(t)
            im -= in[n]*global_cache.sin_vals[idx]; // sin(t)
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

static void fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    const int half_N = N / 2;
    if (N - half_N*2 == 1) {
        dft(in, N, out);
        return;
    }

    float* even = in + N;
    for (int i = 0; i < half_N; ++i) {
        even[i]= in[2*i];
    }
    float* even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float* odd = even;
    for (int i = 0; i < half_N; ++i) {
        odd[i] = in[2*i + 1];
    }
    float* odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        int idx = k * sin_cos_step; // t = 2*M_PI*k/N
        float re = global_cache.cos_vals[idx]; // cos(t)
        float im = -global_cache.sin_vals[idx]; // sin(t)

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + half_N) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + half_N) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

bool whisper_update_filters(const std::string & fname_inp,
                            whisper_filters& filters) {

  printf("%s: loading model from '%s'\n\n", __func__, fname_inp.c_str());

  auto finp = std::ifstream(fname_inp, std::ios::binary);
  if (!finp) {
    fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    finp.read((char *) &magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
      return false;
    }

  }

  whisper_hparams hparams;

  {
    finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
    finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
    finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
    finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
    finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
    finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
    finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
    finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
    finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
    finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
    finp.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

        // const int32_t qntvr_src =    hparams.ftype / GGML_QNT_VERSION_FACTOR;
        // const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

    fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
    fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
    fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
    fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
    fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
    fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
    fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
    fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
    fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
    fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
    fprintf(stderr, "%s: ftype (src)   = %d\n", __func__, hparams.ftype);
    // fprintf(stderr, "%s: qntvr (src)   = %d\n", __func__, qntvr_src);
    // fprintf(stderr, "%s: ftype (dst)   = %d\n", __func__, ftype_dst);
    // fprintf(stderr, "%s: qntvr (dst)   = %d\n", __func__, GGML_QNT_VERSION);

  }

  {
    whisper_filters filters;

    finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
    finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));

    fprintf(stderr, "%s: n_mel   = %d\n", __func__, filters.n_mel);
    fprintf(stderr, "%s: n_fft   = %d\n", __func__, filters.n_fft);

    filters.data.resize(filters.n_mel * filters.n_fft);
    finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
  }


  {
    int32_t n_vocab = 0;
    finp.read ((char *) &n_vocab, sizeof(n_vocab));

    char word[129];

    for (int i = 0; i < n_vocab; i++) {
        uint32_t len;
        finp.read ((char *) &len, sizeof(len));
        word[len] = '\0';
        finp.read ((char *) word, len);    }
  }


  return true;
}


void log_mel_spectrogram_worker_thread(int ith, const float * hann, const std::vector<float> & samples,
                                              int n_samples, int frame_size, int frame_step, int n_threads,
                                              const whisper_filters & filters, whisper_mel & mel) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft = filters.n_fft;
    int i = ith;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    assert(n_fft == 1 + (frame_size / 2));

    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, mel.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 of complex numbers
        // Use pow(fft_out[2 * j + 0], 2) + pow(fft_out[2 * j + 1], 2) causes inference quality problem? Interesting.
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
        }

        // mel spectrogram
        for (int j = 0; j < mel.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft - 3; k += 4) {
                sum +=
                        fft_out[k + 0] * filters.data[j * n_fft + k + 0] +
                        fft_out[k + 1] * filters.data[j * n_fft + k + 1] +
                        fft_out[k + 2] * filters.data[j * n_fft + k + 2] +
                        fft_out[k + 3] * filters.data[j * n_fft + k + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft; k++) {
                sum += fft_out[k] * filters.data[j * n_fft + k];
            }
            sum = log10(std::max(sum, 1e-10));
            mel.data[j * mel.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = log10(1e-10);
    for (; i < mel.n_len; i += n_threads) {
        for (int j = 0; j < mel.n_mel; j++) {
            mel.data[j * mel.n_len + i] = sum;
        }
    }
}

bool log_mel_spectrogram(
  const float * samples,
  const int   n_samples,
  const int   n_mel,
  const int   n_threads,
  const whisper_filters & filters,
  const bool   debug,
  whisper_mel & mel) {


  const float * hann = global_cache.hann_window;

  // Calculate the length of padding
  int64_t stage_1_pad = WHISPER_SAMPLE_RATE * 30;
  int64_t stage_2_pad = WHISPER_N_FFT / 2;

// Initialize a vector and copy data from C array to it.
  std::vector<float> samples_padded;
  samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
  std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);

  // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
  std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);

  // reflective pad up to 200 samples at the beginning of audio
  // clamp the reflected count to the available input so very short audio (n_samples <= stage_2_pad)
  // does not read past the end of `samples`
  const int64_t n_reflect = std::min<int64_t>(stage_2_pad, std::max<int64_t>(0, (int64_t) n_samples - 1));
  std::reverse_copy(samples + 1, samples + 1 + n_reflect, samples_padded.begin() + (stage_2_pad - n_reflect));


  mel.n_mel     = n_mel;
  // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp#L936
  // Calculate number of frames + remove the last frame
  mel.n_len     = (samples_padded.size() - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;
  // Calculate semi-padded sample length to ensure compatibility
  mel.n_len_org = 1 + (n_samples + stage_2_pad - WHISPER_N_FFT) / WHISPER_HOP_LENGTH;
  mel.data.resize(mel.n_mel * mel.n_len);

  {
    std::vector<std::thread> workers(n_threads - 1);
    for (int iw = 0; iw < n_threads - 1; ++iw) {
      workers[iw] = std::thread(
              log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded),
              n_samples + stage_2_pad, WHISPER_N_FFT, WHISPER_HOP_LENGTH, n_threads,
              std::cref(filters), std::ref(mel));
    }

    // main thread
    log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples + stage_2_pad, WHISPER_N_FFT, WHISPER_HOP_LENGTH, n_threads, filters, mel);

    for (int iw = 0; iw < n_threads - 1; ++iw) {
        workers[iw].join();
    }
  }

  double mmax = -1e20;
  for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
    if (mel.data[i] > mmax) {
      mmax = mel.data[i];
    }
  }

  mmax -= 8.0;

  for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
    if (mel.data[i] < mmax) {
      mel.data[i] = mmax;
    }

    mel.data[i] = (mel.data[i] + 4.0)/4.0;
  }

  return 0;
}
