#include "llama.cpp/include/llama.h"

#if defined(__GNUC__) || defined(__clang__)
extern int32_t llama_model_n_embd_full(const struct llama_model * model) __attribute__((weak));
#endif

int32_t llama_model_n_embd_full_compat(const struct llama_model * model) {
#if defined(__GNUC__) || defined(__clang__)
    if (llama_model_n_embd_full) {
        return llama_model_n_embd_full(model);
    }
    return llama_model_n_embd(model);
#else
    return llama_model_n_embd_full(model);
#endif
}
