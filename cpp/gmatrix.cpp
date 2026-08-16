// gmatrix: collect the full Gram matrix X^T X of the input activations of every
// linear layer, for reconstruction-based quantization (--ada in quant-studio).
// Same eval-callback mechanism as llama-imatrix, which only stores the diagonal.
//
// usage: gmatrix -m model.gguf -f calib.txt [-o gmatrix.gguf] [--chunks N] [-ngl 99] [...]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f calib.txt [-o gmatrix.gguf] [--chunks N] [-ngl 99]\n", argv[0]);
    LOG("\n");
}

struct GramSite {
    std::vector<float> gram; // d x d, row-major
    int64_t d     = 0;
    int64_t count = 0;       // tokens accumulated
};

// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

class GramCollector {
public:
    void init_backend();
    bool collect(struct ggml_tensor * t, bool ask);
    void save(const std::string & fname, const std::string & dataset, int32_t chunk_count, int32_t chunk_size);
    size_t n_sites() const { return m_sites.size(); }
private:
    void accumulate(GramSite & site, const float * x, int64_t d, int64_t n_tokens);

    std::unordered_map<std::string, GramSite> m_sites;
    std::mutex          m_mutex;
    std::vector<char>   m_staging;
    std::vector<float>  m_partial;
    ggml_backend_t      m_backend = nullptr;
    bool                m_warned_moe = false;
};

void GramCollector::init_backend() {
    m_backend = ggml_backend_init_best();
    GGML_ASSERT(m_backend != nullptr);
    LOG_INF("%s: accumulating gram matrices on %s\n", __func__, ggml_backend_name(m_backend));
}

// site.gram += x x^T for one batch of activations x [d, n_tokens]
void GramCollector::accumulate(GramSite & site, const float * x, int64_t d, int64_t n_tokens) {
    struct ggml_init_params ip = {
        /* .mem_size   = */ 8*ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_tokens);
    struct ggml_tensor * Xt = ggml_cont(ctx, ggml_transpose(ctx, X));
    struct ggml_tensor * P  = ggml_mul_mat(ctx, Xt, Xt);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, P);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, m_backend);
    GGML_ASSERT(buf != nullptr);
    ggml_backend_tensor_set(X, x, 0, d*n_tokens*sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(m_backend, graph) == GGML_STATUS_SUCCESS);

    m_partial.resize(d*d);
    ggml_backend_tensor_get(P, m_partial.data(), 0, d*d*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    float * acc = site.gram.data();
    const float * part = m_partial.data();
    const int n_th = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int i = 0; i < n_th; ++i) {
        const int64_t lo = d*d *  i    / n_th;
        const int64_t hi = d*d * (i+1) / n_th;
        th.emplace_back([=]() { for (int64_t j = lo; j < hi; ++j) acc[j] += part[j]; });
    }
    for (auto & t : th) t.join();
    site.count += n_tokens;
}

bool GramCollector::collect(struct ggml_tensor * t, bool ask) {
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) {
            if (!m_warned_moe) {
                LOG_WRN("%s: skipping %s: indirect (MoE) matmuls are not collected\n", __func__, wname.c_str());
                m_warned_moe = true;
            }
            return false;
        }
        if (t->op != GGML_OP_MUL_MAT) return false;
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
        if (wname.substr(0, 4) != "blk." || !ggml_is_contiguous(src1)) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    const int64_t d        = src1->ne[0];
    const int64_t n_tokens = ggml_nrows(src1);

    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);
    const float * data;
    if (is_host) {
        data = (const float *) src1->data;
    } else {
        m_staging.resize(ggml_nbytes(src1));
        ggml_backend_tensor_get(src1, m_staging.data(), 0, ggml_nbytes(src1));
        data = (const float *) m_staging.data();
    }

    auto & site = m_sites[wname];
    if (site.gram.empty()) {
        site.d = d;
        site.gram.assign((size_t) d*d, 0.0f);
    }
    GGML_ASSERT(site.d == d);
    accumulate(site, data, d, n_tokens);

    return true;
}

void GramCollector::save(const std::string & fname, const std::string & dataset, int32_t chunk_count, int32_t chunk_size) {
    std::vector<std::string> names;
    names.reserve(m_sites.size());
    for (const auto & kv : m_sites) names.push_back(kv.first);
    std::sort(names.begin(), names.end());

    struct ggml_init_params ip = {
        /* .mem_size   = */ (2*names.size() + 2)*ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    gguf_set_val_str(ctx_gguf, "general.type", "gmatrix");
    gguf_set_val_str(ctx_gguf, "gmatrix.dataset", dataset.c_str());
    gguf_set_val_u32(ctx_gguf, "gmatrix.chunk_count", chunk_count);
    gguf_set_val_u32(ctx_gguf, "gmatrix.chunk_size", chunk_size);

    std::vector<float> counts(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        auto & site = m_sites.at(names[i]);
        counts[i] = (float) site.count;

        struct ggml_tensor * gram = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, site.d, site.d);
        struct ggml_tensor * cnt  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_format_name(gram, "%s.gram", names[i].c_str());
        ggml_format_name(cnt, "%s.counts", names[i].c_str());
        gram->data = site.gram.data();
        cnt->data  = &counts[i];
        gguf_add_tensor(ctx_gguf, gram);
        gguf_add_tensor(ctx_gguf, cnt);
    }

    LOG_INF("%s: writing %zu gram matrices to %s\n", __func__, names.size(), fname.c_str());
    if (!gguf_write_to_file(ctx_gguf, fname.c_str(), false)) {
        LOG_ERR("%s: failed to write %s\n", __func__, fname.c_str());
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

static GramCollector g_collector;

static bool g_collect_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    return g_collector.collect(t, ask);
}

// returns the number of processed chunks, or -1 on failure
static int run_calibration(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    LOG_INF("%s: tokenizing the input ..\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens for a context of %d tokens, got %zu\n", __func__, 2*n_ctx, n_ctx, tokens.size());
        return -1;
    }

    const int n_chunk_max = tokens.size() / n_ctx;
    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = std::min(params.n_batch, n_ctx);
    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    LOG_INF("%s: computing over %d chunks, n_ctx=%d, batch_size=%d\n", __func__, n_chunk, n_ctx, n_batch);

    for (int i = 0; i < n_chunk; ++i) {
        const int start = i * n_ctx;
        const auto t_start = std::chrono::high_resolution_clock::now();

        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(n_ctx - j * n_batch, n_batch);

            common_batch_clear(batch);
            for (int k = 0; k < batch_size; ++k) {
                llama_token tok = (add_bos && j == 0 && k == 0) ? llama_vocab_bos(vocab) : tokens[batch_start + k];
                common_batch_add(batch, tok, j*n_batch + k, { 0 }, false);
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return -1;
            }
        }

        llama_synchronize(ctx);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float t_chunk = std::chrono::duration<float>(t_end - t_start).count();
        if (i == 0) {
            LOG_INF("%s: %.2f seconds per chunk - ETA %.1f minutes\n", __func__, t_chunk, t_chunk * n_chunk / 60.0f);
        }
        LOG("[%d/%d] %.1fs, %zu sites\n", i + 1, n_chunk, t_chunk, g_collector.n_sites());
    }

    llama_batch_free(batch);
    return n_chunk;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "gmatrix.gguf";
    params.n_ctx = 512;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    params.compute_ppl = false;
    params.warmup = false;

    const int32_t n_ctx = params.n_ctx;
    if (n_ctx <= 0) {
        LOG_ERR("%s: gmatrix requires '--ctx-size' > 0\n", __func__);
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("%s: no calibration text provided (-f some-text.txt)\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = g_collect_cb;
    params.cb_eval_user_data = NULL;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    g_collector.init_backend();

    const int n_chunk = run_calibration(ctx, params, n_ctx);
    if (n_chunk < 0) {
        return 1;
    }

    g_collector.save(params.out_file, params.prompt_file, n_chunk, n_ctx);

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
