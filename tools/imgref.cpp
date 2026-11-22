// imgref MODEL MMPROJ IMAGE OUT: the image's embeddings as llama.cpp's mtmd
// computes them (cpu), raw f32 rows of n_embd; prints the token count
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <cstring>

int main(int, char ** v) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    auto * m = llama_model_load_from_file(v[1], mp);
    auto cp = mtmd_context_params_default();
    cp.use_gpu = cp.warmup = false;
    auto * c = mtmd_init_from_file(v[2], m, cp);
    auto b = mtmd_helper_bitmap_init_from_file(c, v[3], false, {});
    mtmd_input_text t = {mtmd_default_marker(), strlen(mtmd_default_marker()), false, true};
    auto * ch = mtmd_input_chunks_init();
    const mtmd_bitmap * bs[] = {b.bitmap};
    if (mtmd_tokenize(c, ch, &t, bs, 1)) return 1;
    for (size_t i = 0; i < mtmd_input_chunks_size(ch); i++) {
        auto * k = mtmd_input_chunks_get(ch, i);
        size_t n = mtmd_input_chunk_get_n_tokens(k);
        if (mtmd_input_chunk_get_type(k) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            auto * id = mtmd_input_chunk_get_tokens_text(k, &n);
            for (size_t j = 0; j < n; j++) printf("text %d\n", id[j]);
            continue;
        }
        if (mtmd_encode_chunk(c, k)) return 2;
        printf("image %zu\n", n);
        auto * f = fopen(v[4], "wb");
        fwrite(mtmd_get_output_embd(c), 4, n * 2304, f);
        fclose(f);
    }
}
