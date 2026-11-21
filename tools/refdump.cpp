// reference dumps from llama.cpp for the inkling engine; see tools/README.md
// refdump MODEL PROMPT.txt OUTDIR [n_gen=64] [n_batch=256] [n_dump=2] [n_logits=65]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
using namespace std;

static string dir;      // step dir being dumped, empty = off
static FILE * idx;
static map<string, int> seen;

static bool cb(ggml_tensor * t, bool ask, void *) {
    // the scheduler names every node; keep the cb() names only, unless ALL is set
    static const bool all = getenv("ALL");
    if (dir.empty() || !t->name[0] || !ggml_is_contiguous(t) || ggml_nbytes(t) == 0 ||
        (!all && (strstr(t->name, " (") || !strncmp(t->name, "node_", 5) || !strncmp(t->name, "leaf_", 5)))) return !ask;
    if (ask) return true;
    string n = t->name;
    for (auto & c : n) if (c == '/' || c == ' ') c = '_';
    int k = seen[n]++;
    if (k) n += "#" + to_string(k);
    vector<char> d(ggml_nbytes(t));
    ggml_backend_tensor_get(t, d.data(), 0, d.size());
    string f = n + "." + ggml_type_name(t->type);
    FILE * o = fopen((dir + "/" + f).c_str(), "wb");
    fwrite(d.data(), 1, d.size(), o);
    fclose(o);
    fprintf(idx, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", f.c_str(), ggml_op_name(t->op),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int c, char ** v) {
    if (c < 4) return fprintf(stderr, "usage: see header\n"), 1;
    auto arg = [&](int i, int d) { return c > i ? atoi(v[i]) : d; };
    const string out = v[3];
    const int n_gen = arg(4, 64), n_batch = arg(5, 256), n_dump = arg(6, 2), n_logits = arg(7, 65);

    FILE * f = fopen(v[2], "rb");
    string text;
    for (int ch; (ch = fgetc(f)) != EOF;) text += ch;
    fclose(f);

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(v[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    vector<llama_token> tok(text.size() + 8);
    tok.resize(llama_tokenize(vocab, text.data(), text.size(), tok.data(), tok.size(), true, false));

    auto cp = llama_context_default_params();
    cp.n_ctx = (tok.size() + n_gen + 256 + 255)/256*256;
    cp.n_batch = cp.n_ubatch = n_batch;
    // FA=0: dense attention in f32, a reference without the f16 kernel
    cp.flash_attn_type = getenv("FA") ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = cb;
    llama_context * ctx = llama_init_from_model(model, cp);

    system(("mkdir -p " + out).c_str());
    auto ids = [&](const char * name, const vector<llama_token> & t) {
        FILE * o = fopen((out + "/" + name).c_str(), "w");
        for (auto x : t) fprintf(o, "%d\n", x);
        fclose(o);
    };
    ids("tokens.txt", tok);

    FILE * lo = fopen((out + "/logits.f32").c_str(), "wb");
    vector<llama_token> gen;
    int n_lo = 0;
    auto next = [&] {
        const float * l = llama_get_logits_ith(ctx, -1);
        if (n_lo++ < n_logits) fwrite(l, 4, n_vocab, lo);
        int b = 0;
        for (int i = 1; i < n_vocab; ++i) if (l[i] > l[b]) b = i;
        gen.push_back(b);
    };
    for (size_t i = 0; i < tok.size(); i += n_batch) {
        if (llama_decode(ctx, llama_batch_get_one(tok.data() + i, min<size_t>(n_batch, tok.size() - i)))) return 2;
    }
    next();
    for (int s = 0; s < n_gen; ++s) {
        if (s < n_dump) {
            char b[32];
            snprintf(b, sizeof b, "/s%04d", s);
            dir = out + b;
            system(("mkdir -p " + dir).c_str());
            idx = fopen((dir + "/index.tsv").c_str(), "w");
            seen.clear();
        }
        if (llama_decode(ctx, llama_batch_get_one(&gen.back(), 1))) return 2;
        if (!dir.empty()) fclose(idx), dir.clear();
        next();
    }
    gen.pop_back();
    fclose(lo);
    ids("gen.txt", gen);
    llama_free(ctx);
    llama_model_free(model);
    fprintf(stderr, "refdump: %zu prompt tokens, %d generated, n_ctx %u, n_batch %d\n", tok.size(), n_gen, cp.n_ctx, n_batch);
    return 0;
}
