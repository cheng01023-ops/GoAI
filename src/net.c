/* net.c - tiny policy/value CNN: two 3x3 conv layers, policy head, value head.
 *
 *   input  P x N x N            (P = 4 feature planes)
 *   conv1  3x3  P -> C          ReLU
 *   conv2  3x3  C -> C          ReLU
 *   policy: conv1x1 C -> 2 planes -> FC (2*N*N) -> (N*N+1) logits -> softmax
 *   value : conv1x1 C -> 1 plane  -> FC (N*N) -> H -> 1 -> tanh
 *
 * Everything (forward AND backward) is implemented by hand with plain loops.
 */
#include "net.h"
#include "compat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void net_config_default(int size, int *planes, int *channels, int *vhidden) {
    if (planes)   *planes = NET_FEATURE_PLANES;
    if (vhidden)  *vhidden = 64;
    if (channels) *channels = (size >= 13) ? 32 : 16;
}

static void layout(Net *net) {
    int o = 0;
    const int nn = net->nn, P = net->planes, C = net->channels, H = net->vhidden;
#define LAY(field, len)                                                          \
    do {                                                                         \
        net->off_##field = o;                                                    \
        net->len_##field = (len);                                                \
        o += (len);                                                              \
    } while (0)
    LAY(c1w, C * P * 9);   LAY(c1b, C);
    LAY(c2w, C * C * 9);   LAY(c2b, C);
    LAY(pw, 2 * C);        LAY(pb, 2);
    LAY(pfcw, (nn + 1) * 2 * nn); LAY(pfcb, nn + 1);
    LAY(vw, C);            LAY(vb, 1);
    LAY(vfc1w, H * nn);    LAY(vfc1b, H);
    LAY(vfc2w, H);         LAY(vfc2b, 1);
#undef LAY
    net->n_params = o;
}

void net_init(Net *net, int size, int planes, int channels, int vhidden, uint64_t seed) {
    memset(net, 0, sizeof(*net));
    net->size = size;
    net->nn = size * size;
    net->planes = planes;
    net->channels = channels;
    net->vhidden = vhidden;
    layout(net);
    net->params = (float *)calloc((size_t)net->n_params, sizeof(float));
    if (!net->params) { fprintf(stderr, "net_init: out of memory\n"); exit(1); }

    Rng rng;
    rng_seed(&rng, seed ? seed : 0x9E3779B9u);
    float *p = net->params;
    const double s1 = sqrt(2.0 / (planes * 9.0));
    const double s2 = sqrt(2.0 / (channels * 9.0));
    const double sp = sqrt(2.0 / (2.0 * net->nn));
    const double sv = sqrt(2.0 / (double)net->nn);
    const double so = 1.0 / sqrt((double)vhidden);
    for (int i = 0; i < net->len_c1w; i++)    p[net->off_c1w + i]    = (float)(rng_normal(&rng) * s1);
    for (int i = 0; i < net->len_c2w; i++)    p[net->off_c2w + i]    = (float)(rng_normal(&rng) * s2);
    for (int i = 0; i < net->len_pw; i++)     p[net->off_pw + i]     = (float)(rng_normal(&rng) * 0.1);
    for (int i = 0; i < net->len_pfcw; i++)   p[net->off_pfcw + i]   = (float)(rng_normal(&rng) * sp * 0.3);
    for (int i = 0; i < net->len_vw; i++)     p[net->off_vw + i]     = (float)(rng_normal(&rng) * 0.1);
    for (int i = 0; i < net->len_vfc1w; i++)  p[net->off_vfc1w + i]  = (float)(rng_normal(&rng) * sv);
    for (int i = 0; i < net->len_vfc2w; i++)  p[net->off_vfc2w + i]  = (float)(rng_normal(&rng) * so);
}

void net_free(Net *net) { free(net->params); net->params = NULL; net->n_params = 0; }

int net_param_count(const Net *net) { return net->n_params; }

float net_param_norm(const Net *net) {
    double s = 0;
    for (int i = 0; i < net->n_params; i++) s += (double)net->params[i] * net->params[i];
    return (float)sqrt(s);
}

void net_zero_grad(const Net *net, float *grad) {
    memset(grad, 0, (size_t)net->n_params * sizeof(float));
}

static float *alloc_floats(size_t n) {
    float *p = (float *)calloc(n ? n : 1, sizeof(float));
    if (!p) { fprintf(stderr, "net: out of memory\n"); exit(1); }
    return p;
}

void net_cache_init(const Net *net, NetCache *c) {
    const size_t nn = (size_t)net->nn, C = (size_t)net->channels, P = (size_t)net->planes, H = (size_t)net->vhidden;
    memset(c, 0, sizeof(*c));
    c->x      = alloc_floats(P * nn);
    c->z1     = alloc_floats(C * nn);
    c->h1     = alloc_floats(C * nn);
    c->z2     = alloc_floats(C * nn);
    c->h2     = alloc_floats(C * nn);
    c->zp     = alloc_floats(2 * nn);
    c->logits = alloc_floats(nn + 1);
    c->probs  = alloc_floats(nn + 1);
    c->zv     = alloc_floats(nn);
    c->za     = alloc_floats(H);
    c->ha     = alloc_floats(H);
    c->dx     = alloc_floats(P * nn);
    c->d_h2   = alloc_floats(C * nn);
    c->d_pre2 = alloc_floats(C * nn);
    c->d_h1   = alloc_floats(C * nn);
    c->d_pre1 = alloc_floats(C * nn);
    c->d_zp   = alloc_floats(2 * nn);
    c->d_zv   = alloc_floats(nn);
    c->d_ha   = alloc_floats(H);
    c->g_scratch = alloc_floats((size_t)net->n_params);
}

void net_cache_free(NetCache *c) {
    free(c->x); free(c->z1); free(c->h1); free(c->z2); free(c->h2); free(c->zp);
    free(c->logits); free(c->probs); free(c->zv); free(c->za); free(c->ha); free(c->dx);
    free(c->d_h2); free(c->d_pre2); free(c->d_h1); free(c->d_pre1);
    free(c->d_zp); free(c->d_zv); free(c->d_ha); free(c->g_scratch);
    memset(c, 0, sizeof(*c));
}

/* ---------------- layers: forward ---------------- */

static void conv3x3_fwd(const float *w, const float *bias, const float *in, int cin, int cout,
                        int n, float *pre, float *act) {
    /* 优化版 v2：把 3x3 卷积拆成 9 个 tap，每个 tap 对整行做 axpy。
       好处：内层循环对 x 连续访存，编译器可以自动向量化（NEON 4 宽），
       而且每个 tap 的内层是纯 float 乘加，没有分支。
       累加平面 n^2 个 float（13 路 676 字节）常驻 L1。 */
    const int nn = n * n;
    for (int co = 0; co < cout; co++) {
        float *acc = pre + (size_t)co * nn;
        const float b = bias[co];
        for (int i = 0; i < nn; i++) acc[i] = b;
        for (int ci = 0; ci < cin; ci++) {
            const float *k = w + ((size_t)co * cin + ci) * 9;
            const float *src = in + (size_t)ci * nn;
            for (int dy = -1; dy <= 1; dy++) {
                const int ylo = dy < 0 ? -dy : 0;
                const int yhi = dy > 0 ? n - 1 - dy : n - 1;
                for (int dx = -1; dx <= 1; dx++) {
                    const float kv = k[(dy + 1) * 3 + (dx + 1)];
                    const int xlo = dx < 0 ? -dx : 0;
                    const int xhi = dx > 0 ? n - 1 - dx : n - 1;
                    if (xlo > xhi) continue;
                    for (int y = ylo; y <= yhi; y++) {
                        const float *srow = src + (size_t)(y + dy) * n + dx;
                        float *arow = acc + (size_t)y * n;
                        for (int x = xlo; x <= xhi; x++)
                            arow[x] += kv * srow[x];
                    }
                }
            }
        }
        float *a = act + (size_t)co * nn;
        for (int i = 0; i < nn; i++) { const float s = acc[i]; a[i] = s > 0.0f ? s : 0.0f; }
    }
}

static void conv1x1_fwd(const float *w, const float *bias, const float *in, int cin, int cout,
                        int n, float *out) {
    const int nn = n * n;
    for (int co = 0; co < cout; co++) {
        const float b = bias[co];
        const float *wp = w + (size_t)co * cin;
        float *op = out + (size_t)co * nn;
        for (int p = 0; p < nn; p++) op[p] = b;
        for (int ci = 0; ci < cin; ci++) {
            const float wv = wp[ci];
            const float *ip = in + (size_t)ci * nn;
            for (int p = 0; p < nn; p++) op[p] += wv * ip[p];
        }
    }
}

void softmax_inplace(const float *logits, float *probs, int n) {
    float m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { probs[i] = (float)exp((double)(logits[i] - m)); sum += probs[i]; }
    const float inv = (float)(1.0 / (sum > 1e-300 ? sum : 1e-300));
    for (int i = 0; i < n; i++) probs[i] *= inv;
}

static float tanh_f(float x) { return tanhf(x); }

void net_forward(const Net *net, NetCache *c, const float *x, float *policy, float *value) {
    const int n = net->size, nn = net->nn, P = net->planes, C = net->channels, H = net->vhidden;
    const float *w = net->params;

    memcpy(c->x, x, (size_t)P * nn * sizeof(float));
    conv3x3_fwd(w + net->off_c1w, w + net->off_c1b, x, P, C, n, c->z1, c->h1);
    conv3x3_fwd(w + net->off_c2w, w + net->off_c2b, c->h1, C, C, n, c->z2, c->h2);
    conv1x1_fwd(w + net->off_pw, w + net->off_pb, c->h2, C, 2, n, c->zp);
    conv1x1_fwd(w + net->off_vw, w + net->off_vb, c->h2, C, 1, n, c->zv);

    const float *pfcw = w + net->off_pfcw;
    const float *pfcb = w + net->off_pfcb;
    for (int m = 0; m <= nn; m++) {
        float s = pfcb[m];
        const float *row = pfcw + (size_t)m * 2 * nn;
        for (int i = 0; i < 2 * nn; i++) s += row[i] * c->zp[i];
        c->logits[m] = s;
    }
    softmax_inplace(c->logits, c->probs, nn + 1);
    if (policy) memcpy(policy, c->probs, (size_t)(nn + 1) * sizeof(float));

    const float *vfc1w = w + net->off_vfc1w;
    const float *vfc1b = w + net->off_vfc1b;
    for (int j = 0; j < H; j++) {
        float s = vfc1b[j];
        const float *row = vfc1w + (size_t)j * nn;
        for (int i = 0; i < nn; i++) s += row[i] * c->zv[i];
        c->za[j] = s;
        c->ha[j] = s > 0.0f ? s : 0.0f;
    }
    float vo = w[net->off_vfc2b];
    for (int j = 0; j < H; j++) vo += w[net->off_vfc2w + j] * c->ha[j];
    c->vo = vo;
    c->v = tanh_f(vo);
    if (value) *value = c->v;
}

/* ---------------- layers: backward ---------------- */

static void conv3x3_bwd(const float *w, const float *pre, const float *dact, const float *in,
                        int cin, int cout, int n, float *gw, float *gb, float *din) {
    for (int co = 0; co < cout; co++) {
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                const size_t oidx = (size_t)co * n * n + y * n + x;
                float d = dact[oidx];
                if (d == 0.0f) continue;
                if (!(pre[oidx] > 0.0f)) continue;          /* ReLU derivative */
                gb[co] += d;
                for (int ci = 0; ci < cin; ci++) {
                    const float *ip = in + (size_t)ci * n * n;
                    float *gwp = gw + ((size_t)co * cin + ci) * 9;
                    const float *wp = w + ((size_t)co * cin + ci) * 9;
                    for (int dy = -1; dy <= 1; dy++) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= n) continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            const int xx = x + dx;
                            if (xx < 0 || xx >= n) continue;
                            gwp[(dy + 1) * 3 + (dx + 1)] += d * ip[yy * n + xx];
                            if (din) din[(size_t)ci * n * n + yy * n + xx] += d * wp[(dy + 1) * 3 + (dx + 1)];
                        }
                    }
                }
            }
        }
    }
}

static void conv1x1_bwd(const float *w, const float *in, const float *dout, int cin, int cout,
                        int n, float *gw, float *gb, float *din) {
    const int nn = n * n;
    for (int co = 0; co < cout; co++) {
        const float *dop = dout + (size_t)co * nn;
        float *gwp = gw + (size_t)co * cin;
        double sum = 0;
        for (int p = 0; p < nn; p++) sum += dop[p];
        gb[co] += (float)sum;
        for (int ci = 0; ci < cin; ci++) {
            const float wv = w[(size_t)co * cin + ci];
            const float *ip = in + (size_t)ci * nn;
            double acc = 0;
            for (int p = 0; p < nn; p++) acc += (double)dop[p] * ip[p];
            gwp[ci] += (float)acc;
            if (din) {
                float *dp = din + (size_t)ci * nn;
                for (int p = 0; p < nn; p++) dp[p] += dop[p] * wv;
            }
        }
    }
}

float net_backward(const Net *net, NetCache *c, const float *pi, float z, float *grad,
                   float *out_policy_loss, float *out_value_loss) {
    const int nn = net->nn, P = net->planes, C = net->channels, H = net->vhidden;
    float *g = grad;
    if (!g) { g = c->g_scratch; memset(g, 0, (size_t)net->n_params * sizeof(float)); }

    /* ---- loss and its derivatives ---- */
    double ce = 0.0;
    for (int m = 0; m <= nn; m++) {
        const double p = c->probs[m] > 1e-12f ? (double)c->probs[m] : 1e-12;
        ce -= (double)pi[m] * log(p);
    }
    const double dv = 2.0 * ((double)c->v - (double)z);         /* dL/dv  */
    const double mse = (double)(c->v - z) * (double)(c->v - z);
    if (out_policy_loss) *out_policy_loss = (float)ce;
    if (out_value_loss)  *out_value_loss  = (float)mse;

    /* ---- value head ---- */
    const float dvo = (float)(dv * (1.0 - (double)c->v * (double)c->v));
    {
        float *gvfc2w = g + net->off_vfc2w, *gvfc2b = g + net->off_vfc2b;
        gvfc2b[0] += dvo;
        for (int j = 0; j < H; j++) {
            gvfc2w[j] += dvo * c->ha[j];
            c->d_ha[j] = dvo * net->params[net->off_vfc2w + j];
        }
        float *gvfc1w = g + net->off_vfc1w, *gvfc1b = g + net->off_vfc1b;
        memset(c->d_zv, 0, (size_t)nn * sizeof(float));
        for (int j = 0; j < H; j++) {
            if (!(c->za[j] > 0.0f)) continue;
            const float dza = c->d_ha[j];
            if (dza == 0.0f) continue;
            gvfc1b[j] += dza;
            float *grow = gvfc1w + (size_t)j * nn;
            const float *wrow = net->params + net->off_vfc1w + (size_t)j * nn;
            for (int i = 0; i < nn; i++) {
                grow[i] += dza * c->zv[i];
                c->d_zv[i] += dza * wrow[i];
            }
        }
        memset(c->d_h2, 0, (size_t)C * nn * sizeof(float));
        conv1x1_bwd(net->params + net->off_vw, c->h2, c->d_zv, C, 1, net->size,
                    g + net->off_vw, g + net->off_vb, c->d_h2);
    }

    /* ---- policy head ---- */
    {
        float *gpfcw = g + net->off_pfcw, *gpfcb = g + net->off_pfcb;
        memset(c->d_zp, 0, (size_t)2 * nn * sizeof(float));
        for (int m = 0; m <= nn; m++) {
            const float dl = c->probs[m] - pi[m];
            gpfcb[m] += dl;
            float *grow = gpfcw + (size_t)m * 2 * nn;
            const float *wrow = net->params + net->off_pfcw + (size_t)m * 2 * nn;
            for (int i = 0; i < 2 * nn; i++) {
                grow[i] += dl * c->zp[i];
                c->d_zp[i] += dl * wrow[i];
            }
        }
        conv1x1_bwd(net->params + net->off_pw, c->h2, c->d_zp, C, 2, net->size,
                    g + net->off_pw, g + net->off_pb, c->d_h2);
    }

    /* ---- conv2 ---- */
    memset(c->d_h1, 0, (size_t)C * nn * sizeof(float));
    conv3x3_bwd(net->params + net->off_c2w, c->z2, c->d_h2, c->h1, C, C, net->size,
                g + net->off_c2w, g + net->off_c2b, c->d_h1);

    /* ---- conv1 ---- */
    memset(c->dx, 0, (size_t)P * nn * sizeof(float));
    conv3x3_bwd(net->params + net->off_c1w, c->z1, c->d_h1, c->x, P, C, net->size,
                g + net->off_c1w, g + net->off_c1b, c->dx);

    return (float)(ce + mse);
}

void net_adam(Net *net, float *grad, float *m, float *v, int step, float lr,
              float weight_decay, float clip_norm) {
    float *p = net->params;
    const int n = net->n_params;
    double sq = 0;
    for (int i = 0; i < n; i++) sq += (double)grad[i] * grad[i];
    const double norm = sqrt(sq);
    double scale = 1.0;
    if (clip_norm > 0.0f && norm > (double)clip_norm) scale = (double)clip_norm / (norm + 1e-12);

    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float bc1 = 1.0f - powf(b1, (float)step);
    const float bc2 = 1.0f - powf(b2, (float)step);
    for (int i = 0; i < n; i++) {
        const float gi = (float)(grad[i] * scale);
        m[i] = b1 * m[i] + (1.0f - b1) * gi;
        v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        const float mh = m[i] / bc1;
        const float vh = v[i] / bc2;
        p[i] -= lr * (mh / (sqrtf(vh) + eps) + weight_decay * p[i]);
    }
}

/* ---------------- serialisation ---------------- */

typedef struct {
    uint32_t magic;
    int32_t  size, planes, channels, vhidden, n_params;
} NetHeader;

bool net_save(const Net *net, const char *path) {
    /* 先写临时文件再改名：这样正在对弈的程序读到的永远是完整文件
       （训练器每轮覆写 latest.bin，若直接写会有一瞬间是半截文件） */
    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.tmp%ld", path, goai_pid());
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    NetHeader h;
    h.magic = NET_MAGIC;
    h.size = net->size; h.planes = net->planes; h.channels = net->channels;
    h.vhidden = net->vhidden; h.n_params = net->n_params;
    const bool ok = fwrite(&h, sizeof(h), 1, f) == 1 &&
                    fwrite(net->params, sizeof(float), (size_t)net->n_params, f) == (size_t)net->n_params;
    fflush(f);
    fclose(f);
    if (!ok) { remove(tmp); return false; }
    if (rename(tmp, path) != 0) { remove(tmp); return false; }
    return true;
}

/* 从内存加载（权重编译进程序时用，不需要外部文件） */
bool net_load_mem(Net *net, const void *data, size_t len) {
    if (!data || len < sizeof(NetHeader)) return false;
    NetHeader h;
    memcpy(&h, data, sizeof(h));
    if (h.magic != NET_MAGIC) return false;
    if (h.n_params <= 0 || len < sizeof(NetHeader) + (size_t)h.n_params * sizeof(float)) {
        fprintf(stderr, "net_load_mem: 内置权重长度不对（%zu 字节，需要 %zu）\n",
                len, sizeof(NetHeader) + (size_t)h.n_params * sizeof(float));
        return false;
    }
    if (net->params == NULL || net->size != h.size || net->planes != h.planes ||
        net->channels != h.channels || net->vhidden != h.vhidden || net->n_params != h.n_params) {
        net_free(net);
        net_init(net, h.size, h.planes, h.channels, h.vhidden, 1);
    }
    memcpy(net->params, (const unsigned char *)data + sizeof(NetHeader),
           (size_t)net->n_params * sizeof(float));
    return true;
}

bool net_load(Net *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    NetHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != NET_MAGIC) { fclose(f); return false; }
    if (net->params == NULL || net->size != h.size || net->planes != h.planes ||
        net->channels != h.channels || net->vhidden != h.vhidden || net->n_params != h.n_params) {
        net_free(net);
        net_init(net, h.size, h.planes, h.channels, h.vhidden, 1);
    }
    const size_t got = fread(net->params, sizeof(float), (size_t)net->n_params, f);
    fclose(f);
    if (got != (size_t)net->n_params) {
        fprintf(stderr, "net_load: %s 数据不完整（读到 %zu/%d 个参数），忽略这次加载\n",
                path, got, net->n_params);
        return false;
    }
    return true;
}

void net_features(const Board *b, float *x) {
    const int nn = b->size * b->size;
    const int me = b->to_move, opp = 3 - me;
    for (int p = 0; p < nn; p++) {
        x[p] = b->cell[p] == me ? 1.0f : 0.0f;
        x[nn + p] = b->cell[p] == opp ? 1.0f : 0.0f;
        x[2 * nn + p] = (p == b->ko) ? 1.0f : 0.0f;
        x[3 * nn + p] = 1.0f;
    }
}
