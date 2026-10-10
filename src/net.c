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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif
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

/* 参数总量：与 layout() 必须严格一致（加载旧文件时用它交叉核对 flags 解释） */
static int params_for(int nn, int planes, int channels, int vhidden, int blocks,
                      int own_head, int vdist) {
    const int K = vdist ? NET_VALUE_BUCKETS : 1;
    int o = 0;
    o += channels * planes * 9 + channels;                        /* conv1 */
    o += channels * channels * 9 + channels;                      /* conv2 */
    o += blocks * (2 * channels * channels * 9 + 2 * channels);   /* 残差块 */
    o += 2 * channels + 2;                                        /* 策略头 1x1 */
    o += (nn + 1) * 2 * nn + (nn + 1);                            /* 策略全连接 */
    o += channels + 1;                                            /* 价值头 1x1 */
    o += vhidden * nn + vhidden;                                  /* 价值全连接 1 */
    o += vhidden * K + K;                                         /* 价值全连接 2（K 桶） */
    if (own_head) o += channels + 1;                              /* 领地头 1x1 */
    return o;
}

static void layout(Net *net) {
    int o = 0;
    const int nn = net->nn, P = net->planes, C = net->channels, H = net->vhidden;
    net->flags = NET_FLAG_BLOCKS | (net->own_head ? NET_FLAG_OWN : 0) |
                 (net->vdist ? NET_FLAG_VDIST : 0) | ((uint32_t)net->blocks << 8);
    net->nbuckets = net->vdist ? NET_VALUE_BUCKETS : 1;
#define LAY(field, len)                                                          \
    do {                                                                         \
        net->off_##field = o;                                                    \
        net->len_##field = (len);                                                \
        o += (len);                                                              \
    } while (0)
    LAY(c1w, C * P * 9);   LAY(c1b, C);
    LAY(c2w, C * C * 9);   LAY(c2b, C);
    /* 每个残差块：convA(w,b) + convB(w,b) */
    net->blk_stride = 2 * C * C * 9 + 2 * C;
    LAY(bw, net->blocks * net->blk_stride);
    LAY(pw, 2 * C);        LAY(pb, 2);
    LAY(pfcw, (nn + 1) * 2 * nn); LAY(pfcb, nn + 1);
    LAY(vw, C);            LAY(vb, 1);
    LAY(vfc1w, H * nn);    LAY(vfc1b, H);
    LAY(vfc2w, H * net->nbuckets); LAY(vfc2b, net->nbuckets);
    /* 领地头放最后：这样老文件的参数前缀能原样搬进新网络（net_copy_shared） */
    if (net->own_head) { LAY(ow, C); LAY(ob, 1); }
    else { net->off_ow = o; net->off_ob = o; net->len_ow = 0; net->len_ob = 0; }
#undef LAY
    net->n_params = o;
}

/* 值分布的桶心：33 个桶均匀覆盖 [-1, +1] */
float net_value_bucket_center(int k) {
    if (k < 0) k = 0;
    if (k > NET_VALUE_BUCKETS - 1) k = NET_VALUE_BUCKETS - 1;
    return -1.0f + 2.0f * (float)k / (float)(NET_VALUE_BUCKETS - 1);
}

int net_value_bucket_index(float z) {
    if (z < -1.0f) z = -1.0f;
    if (z > 1.0f) z = 1.0f;
    const float t = (z + 1.0f) * 0.5f * (float)(NET_VALUE_BUCKETS - 1);
    int k = (int)(t + 0.5f);
    if (k < 0) k = 0;
    if (k > NET_VALUE_BUCKETS - 1) k = NET_VALUE_BUCKETS - 1;
    return k;
}

void net_arch_str(const Net *net, char *buf, size_t n) {
    char tails[64];
    tails[0] = 0;
    if (net->blocks > 0) snprintf(tails + strlen(tails), sizeof(tails) - strlen(tails),
                                  " %dblk", net->blocks);
    if (net->own_head)   snprintf(tails + strlen(tails), sizeof(tails) - strlen(tails), " own");
    if (net->vdist)      snprintf(tails + strlen(tails), sizeof(tails) - strlen(tails),
                                  " vdist%d", NET_VALUE_BUCKETS);
    snprintf(buf, n, "%dch%s", net->channels, tails);
}

void net_init(Net *net, int size, int planes, int channels, int vhidden, uint64_t seed) {
    net_init_ex(net, size, planes, channels, vhidden, 0, seed);
}

void net_init_ex(Net *net, int size, int planes, int channels, int vhidden,
                 int blocks, uint64_t seed) {
    net_init_full(net, size, planes, channels, vhidden, blocks, 0, 0, seed);
}

void net_init_full(Net *net, int size, int planes, int channels, int vhidden,
                   int blocks, int own_head, int vdist, uint64_t seed) {
    memset(net, 0, sizeof(*net));
    net->blocks = blocks < 0 ? 0 : blocks;
    net->own_head = own_head ? 1 : 0;
    net->vdist = vdist ? 1 : 0;
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
    for (int i = 0; i < net->len_bw; i++)     p[net->off_bw + i]     = (float)(rng_normal(&rng) * s2);
    /* 残差块零初始化：把每块第二个卷积（含偏置）置零，使块初始就是恒等映射。
       这是 ResNet / AlphaZero 的标准做法 —— 否则随机残差会破坏预训练特征：
       实测未做零初始化时，热启动后对随机胜率从 90% 掉到 12%，训练也随之发散。 */
    for (int b = 0; b < net->blocks; b++) {
        float *base = p + net->off_bw + (size_t)b * net->blk_stride;
        const int cw = net->channels * net->channels * 9;
        for (int i = 0; i < cw; i++) base[cw + net->channels + i] = 0.0f;              /* wB = 0 */
        for (int i = 0; i < net->channels; i++) base[2 * cw + net->channels + i] = 0.0f; /* bB = 0 */
    }
    for (int i = 0; i < net->len_pw; i++)     p[net->off_pw + i]     = (float)(rng_normal(&rng) * 0.1);
    for (int i = 0; i < net->len_pfcw; i++)   p[net->off_pfcw + i]   = (float)(rng_normal(&rng) * sp * 0.3);
    for (int i = 0; i < net->len_vw; i++)     p[net->off_vw + i]     = (float)(rng_normal(&rng) * 0.1);
    for (int i = 0; i < net->len_vfc1w; i++)  p[net->off_vfc1w + i]  = (float)(rng_normal(&rng) * sv);
    for (int i = 0; i < net->len_vfc2w; i++)  p[net->off_vfc2w + i]  = (float)(rng_normal(&rng) * so);
    /* 值分布头：K 个桶的 logits 初始化得接近 0 -> softmax 近似均匀 -> 搜索值约 0，
       和原来标量头"开始时 v 无信息"的行为一致。 */
    for (int i = 0; i < net->len_vfc2b; i++)  p[net->off_vfc2b + i]  = 0.0f;
    /* 领地头：小权重（1x1 conv C->1），让初始 logits 贴近 0，sigmoid 不饱和。
       OWN_WEIGHT 会把它放大成有效的辅助梯度，所以要压得比策略头更小。 */
    if (net->own_head) {
        const double sow = 0.1 / sqrt((double)channels);
        for (int i = 0; i < net->len_ow; i++) p[net->off_ow + i] = (float)(rng_normal(&rng) * sow);
        for (int i = 0; i < net->len_ob; i++) p[net->off_ob + i] = 0.0f;
    }
}

void net_free(Net *net) { free(net->params); net->params = NULL; net->n_params = 0; }

/* 把 src 中形状相同的参数段拷贝到 dst。用于"给已训练的网络加残差块"：
   卷积层/头部权重原样搬过去，新加的残差块保持随机初始化。 */
void net_copy_shared(Net *dst, const Net *src) {
    if (dst->size != src->size || dst->planes != src->planes ||
        dst->channels != src->channels || dst->vhidden != src->vhidden) return;
    /* 只在两端长度一致时拷贝：值头从标量(H)换成分布(H*K)时形状不同，不能硬拷
       （新头保持 net_init_full 的随机初始化）。 */
    const struct { int d, s, n, sn; } seg[] = {
        { dst->off_c1w, src->off_c1w, dst->len_c1w, src->len_c1w },
        { dst->off_c1b, src->off_c1b, dst->len_c1b, src->len_c1b },
        { dst->off_c2w, src->off_c2w, dst->len_c2w, src->len_c2w },
        { dst->off_c2b, src->off_c2b, dst->len_c2b, src->len_c2b },
        { dst->off_pw, src->off_pw, dst->len_pw, src->len_pw },
        { dst->off_pb, src->off_pb, dst->len_pb, src->len_pb },
        { dst->off_pfcw, src->off_pfcw, dst->len_pfcw, src->len_pfcw },
        { dst->off_pfcb, src->off_pfcb, dst->len_pfcb, src->len_pfcb },
        { dst->off_vw, src->off_vw, dst->len_vw, src->len_vw },
        { dst->off_vb, src->off_vb, dst->len_vb, src->len_vb },
        { dst->off_vfc1w, src->off_vfc1w, dst->len_vfc1w, src->len_vfc1w },
        { dst->off_vfc1b, src->off_vfc1b, dst->len_vfc1b, src->len_vfc1b },
        { dst->off_vfc2w, src->off_vfc2w, dst->len_vfc2w, src->len_vfc2w },
        { dst->off_vfc2b, src->off_vfc2b, dst->len_vfc2b, src->len_vfc2b },
    };
    for (size_t i = 0; i < sizeof(seg) / sizeof(seg[0]); i++)
        if (seg[i].n > 0 && seg[i].n == seg[i].sn)
            memcpy(dst->params + seg[i].d, src->params + seg[i].s, (size_t)seg[i].n * sizeof(float));
}

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
    c->zo     = alloc_floats(nn);                       /* 领地 logits */
    c->za     = alloc_floats(H);
    c->ha     = alloc_floats(H);
    c->vlogits = alloc_floats(NET_VALUE_BUCKETS);       /* 值分布 */
    c->vprobs  = alloc_floats(NET_VALUE_BUCKETS);
    c->dx     = alloc_floats(P * nn);
    c->d_h2   = alloc_floats(C * nn);
    c->d_pre2 = alloc_floats(C * nn);
    c->d_h1   = alloc_floats(C * nn);
    c->d_pre1 = alloc_floats(C * nn);
    c->d_zp   = alloc_floats(2 * nn);
    c->d_zv   = alloc_floats(nn);
    c->d_zo   = alloc_floats(nn);
    c->d_ha   = alloc_floats(H);
    c->g_scratch = alloc_floats((size_t)net->n_params);
    /* 残差块的前向/反向缓存 */
    if (net->blocks > 0) {
        const size_t nb = (size_t)net->blocks * 4 * net->channels * net->nn;
        c->blk = (float *)calloc(nb, sizeof(float));
        c->dblk = (float *)calloc((size_t)3 * net->channels * net->nn, sizeof(float));
        if (!c->blk || !c->dblk) {
            fprintf(stderr, "net_cache_init: out of memory\n");
            exit(1);
        }
    }
}


void net_cache_free(NetCache *c) {
    free(c->x); free(c->z1); free(c->h1); free(c->z2); free(c->h2); free(c->zp);
    free(c->logits); free(c->probs); free(c->zv); free(c->zo); free(c->za); free(c->ha); free(c->dx);
    free(c->vlogits); free(c->vprobs);
    free(c->d_h2); free(c->d_pre2); free(c->d_h1); free(c->d_pre1);
    free(c->d_zp); free(c->d_zv); free(c->d_zo); free(c->d_ha); free(c->g_scratch);
    free(c->blk); c->blk = NULL;
    free(c->dblk); c->dblk = NULL;
    memset(c, 0, sizeof(*c));
}

/* ---------------- layers: forward ---------------- */

/* 行内 axpy：acc[i] += kv * src[i]。ARM 上用手写 NEON（4 宽 FMA），   其他平台退回标量循环（编译器同样能向量化）。 */
static inline void axpy_row(float *acc, const float *src, float kv, int len) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float32x4_t kvv = vdupq_n_f32(kv);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        const float32x4_t a = vld1q_f32(acc + i);
        const float32x4_t b = vld1q_f32(src + i);
        vst1q_f32(acc + i, vfmaq_f32(a, b, kvv));
    }
    for (; i < len; i++) acc[i] += kv * src[i];
#else
    for (int i = 0; i < len; i++) acc[i] += kv * src[i];
#endif
}
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
                        axpy_row(arow + xlo, srow + xlo, kv, xhi - xlo + 1);
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

    /* ---- 残差块：h <- relu(convB(relu(convA(h))) + h) ---- */
    const float *h = c->h2;
    for (int b = 0; b < net->blocks; b++) {
        const float *base = w + net->off_bw + (size_t)b * net->blk_stride;
        float *zA = c->blk + (size_t)b * 4 * C * nn;
        float *hA = zA + C * nn, *zB = hA + C * nn, *hB = zB + C * nn;
        conv3x3_fwd(base, base + C * C * 9, h, C, C, n, zA, hA);
        conv3x3_fwd(base + C * C * 9 + C, base + C * C * 9 + C + C * C * 9,
                    hA, C, C, n, zB, hB);
        for (int i = 0; i < C * nn; i++) {
            const float v = zB[i] + h[i];
            hB[i] = v > 0.0f ? v : 0.0f;
        }
        h = hB;
    }

    conv1x1_fwd(w + net->off_pw, w + net->off_pb, h, C, 2, n, c->zp);
    conv1x1_fwd(w + net->off_vw, w + net->off_vb, h, C, 1, n, c->zv);
    /* 领地头：1x1 conv C->1，每个点位一个 logit（训练时 sigmoid 后算 BCE；
       下棋/搜索用不到这个头） */
    if (net->own_head) conv1x1_fwd(w + net->off_ow, w + net->off_ob, h, C, 1, n, c->zo);

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
    if (net->vdist) {
        /* 值分布头：K 个桶的 logits -> softmax -> 期望值（搜索用期望，训练用交叉熵） */
        const int K = net->nbuckets;
        for (int k = 0; k < K; k++) {
            float s = w[net->off_vfc2b + k];
            for (int j = 0; j < H; j++) s += w[net->off_vfc2w + (size_t)j * K + k] * c->ha[j];
            c->vlogits[k] = s;
        }
        softmax_inplace(c->vlogits, c->vprobs, K);
        double e = 0.0;
        for (int k = 0; k < K; k++) e += (double)c->vprobs[k] * (double)net_value_bucket_center(k);
        c->vo = (float)e;
        c->v = (float)e;
    } else {
        float vo = w[net->off_vfc2b];
        for (int j = 0; j < H; j++) vo += w[net->off_vfc2w + j] * c->ha[j];
        c->vo = vo;
        c->v = tanh_f(vo);
    }
    if (value) *value = c->v;
}

/* ---------------- layers: backward ---------------- */

/* 优化的 3x3 反向：先把 ReLU 掩码应用到梯度上（一次分支判断），
   之后所有内层循环都是无分支、连续访存的（可向量化）：
     · 卷积核梯度 gw：对各 tap 做点积累加
     · 输入梯度 din：对各 tap 做 axpy（复用前向的 NEON 版本）
   gb 用一次求和代替逐像素累加。 */
static void conv3x3_bwd(const float *w, const float *pre, const float *dact, const float *in,
                        int cin, int cout, int n, float *gw, float *gb, float *din) {
    const int nn = n * n;
    static _Thread_local float *dm = NULL;
    static _Thread_local int dm_cap = 0;
    if (nn > dm_cap) {
        free(dm);
        dm = (float *)malloc((size_t)nn * sizeof(float));
        dm_cap = dm ? nn : 0;
        if (!dm) return;
    }
    for (int co = 0; co < cout; co++) {
        const float *prec = pre + (size_t)co * nn;
        const float *dacc = dact + (size_t)co * nn;
        float gbsum = 0.0f;
        for (int i = 0; i < nn; i++) {
            const float v = (prec[i] > 0.0f) ? dacc[i] : 0.0f;
            dm[i] = v;
            gbsum += v;
        }
        gb[co] += gbsum;
        for (int ci = 0; ci < cin; ci++) {
            const float *k = w + ((size_t)co * cin + ci) * 9;
            float *gwp = gw + ((size_t)co * cin + ci) * 9;
            const float *src = in + (size_t)ci * nn;
            float *dout = din ? din + (size_t)ci * nn : NULL;
            for (int dy = -1; dy <= 1; dy++) {
                const int ylo = dy < 0 ? -dy : 0;
                const int yhi = dy > 0 ? n - 1 - dy : n - 1;
                for (int dx = -1; dx <= 1; dx++) {
                    const int tap = (dy + 1) * 3 + (dx + 1);
                    const float kv = k[tap];
                    const int xlo = dx < 0 ? -dx : 0;
                    const int xhi = dx > 0 ? n - 1 - dx : n - 1;
                    if (xlo > xhi) continue;
                    float acc = 0.0f;
                    for (int y = ylo; y <= yhi; y++) {
                        const float *drow = dm + (size_t)y * n;
                        const float *srow = src + (size_t)(y + dy) * n + dx;
                        for (int x = xlo; x <= xhi; x++) acc += drow[x] * srow[x];
                        if (dout) {
                            float *orow = dout + (size_t)(y + dy) * n + dx;
                            axpy_row(orow + xlo, drow + xlo, kv, xhi - xlo + 1);
                        }
                    }
                    gwp[tap] += acc;
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

/* 兼容包装：只有策略 + 价值两个目标 */
float net_backward(const Net *net, NetCache *c, const float *pi, float z, float *grad,
                   float *out_policy_loss, float *out_value_loss) {
    return net_backward_ex(net, c, pi, z, NULL, 0.0f, grad,
                           out_policy_loss, out_value_loss, NULL);
}

float net_backward_ex(const Net *net, NetCache *c, const float *pi, float z,
                      const float *own_target, float own_weight, float *grad,
                      float *out_policy_loss, float *out_value_loss, float *out_own_loss) {
    const int nn = net->nn, P = net->planes, C = net->channels, H = net->vhidden;
    float *g = grad;
    if (!g) { g = c->g_scratch; memset(g, 0, (size_t)net->n_params * sizeof(float)); }

    /* ---- 策略损失 CE(policy, pi) ---- */
    double ce = 0.0;
    for (int m = 0; m <= nn; m++) {
        const double p = c->probs[m] > 1e-12f ? (double)c->probs[m] : 1e-12;
        ce -= (double)pi[m] * log(p);
    }
    if (out_policy_loss) *out_policy_loss = (float)ce;

    /* ---- 价值头：分布走交叉熵，标量走 MSE ---- */
    double vl = 0.0;
    {
        float *gvfc2w = g + net->off_vfc2w, *gvfc2b = g + net->off_vfc2b;
        if (net->vdist) {
            const int K = net->nbuckets;
            const int tgt = net_value_bucket_index(z);     /* z 落入的桶（one-hot 目标） */
            const float pt = c->vprobs[tgt] > 1e-12f ? c->vprobs[tgt] : 1e-12f;
            vl = -log((double)pt);
            memset(c->d_ha, 0, (size_t)H * sizeof(float));
            for (int k = 0; k < K; k++) {
                const float dl = c->vprobs[k] - (k == tgt ? 1.0f : 0.0f);   /* dL/dlogit */
                gvfc2b[k] += dl;
                for (int j = 0; j < H; j++) {
                    gvfc2w[(size_t)j * K + k] += dl * c->ha[j];
                    c->d_ha[j] += dl * net->params[net->off_vfc2w + (size_t)j * K + k];
                }
            }
        } else {
            const double dv = 2.0 * ((double)c->v - (double)z);         /* dL/dv  */
            vl = (double)(c->v - z) * (double)(c->v - z);
            const float dvot = (float)(dv * (1.0 - (double)c->v * (double)c->v));
            gvfc2b[0] += dvot;
            for (int j = 0; j < H; j++) {
                gvfc2w[j] += dvot * c->ha[j];
                c->d_ha[j] = dvot * net->params[net->off_vfc2w + j];
            }
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
    if (out_value_loss) *out_value_loss = (float)vl;

    /* ---- 策略头 ---- */
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

    /* ---- 领地头：BCE（对 sigmoid(logit)）----
       平均到每个点位（/nn），再乘 OWN_WEIGHT；梯度 d_logit = (sigmoid(z) - t)/nn。
       梯度累加到 d_h2，与策略/价值头的梯度相加 ✓ */
    double ol = 0.0;
    if (net->own_head && own_target && own_weight != 0.0f) {
        const float scale = own_weight / (float)nn;
        for (int i = 0; i < nn; i++) {
            const float zi = c->zo[i];
            const float t = own_target[i];
            /* 数值稳定的 BCE：max(z,0) - z*t + log1p(exp(-|z|)) */
            ol += (double)(fmaxf(zi, 0.0f) - zi * t + log1pf(expf(-fabsf(zi))));
            const float p = 1.0f / (1.0f + expf(-zi));      /* sigmoid */
            c->d_zo[i] = (p - t) * scale;
        }
        conv1x1_bwd(net->params + net->off_ow, c->h2, c->d_zo, C, 1, net->size,
                    g + net->off_ow, g + net->off_ob, c->d_h2);
        ol /= (double)nn;
    }
    if (out_own_loss) *out_own_loss = (float)ol;

    /* ---- 残差块反传（从最后一塊往前）---- */
    const float *dcur = c->d_h2;      /* 头部累积的梯度（对最终激活） */
    for (int b = net->blocks - 1; b >= 0; b--) {
        float *zA = c->blk + (size_t)b * 4 * C * nn;
        float *hA = zA + C * nn, *zB = hA + C * nn, *hB = zB + C * nn;
        const float *hin = (b == 0) ? c->h2 : (c->blk + (size_t)(b - 1) * 4 * C * nn + 3 * C * nn);
        float *dhA = c->dblk + (size_t)(b % 3) * C * nn;
        float *din = c->dblk + (size_t)((b + 1) % 3) * C * nn;
        const float *base = net->params + net->off_bw + (size_t)b * net->blk_stride;
        float *gw = g + net->off_bw + (size_t)b * net->blk_stride;
        memset(dhA, 0, (size_t)C * nn * sizeof(float));
        memset(din, 0, (size_t)C * nn * sizeof(float));
        /* convB 的反向：hB 提供 ReLU 掩码（hB>0 <=> 预激活>0）*/
        conv3x3_bwd(base + C * C * 9 + C, hB, dcur, hA, C, C, net->size,
                    gw + C * C * 9 + C, gw + C * C * 9 + C + C * C * 9, dhA);
        /* convA 的反向 */
        conv3x3_bwd(base, zA, dhA, hin, C, C, net->size, gw, gw + C * C * 9, din);
        /* 跳连：h_out = relu(zB + hin) => d(hin) += dcur * (hB>0) */
        for (int i = 0; i < C * nn; i++)
            if (hB[i] > 0.0f) din[i] += dcur[i];
        dcur = din;
    }

    /* ---- conv2 ---- */
    memset(c->d_h1, 0, (size_t)C * nn * sizeof(float));
    conv3x3_bwd(net->params + net->off_c2w, c->z2, dcur, c->h1, C, C, net->size,
                g + net->off_c2w, g + net->off_c2b, c->d_h1);

    /* ---- conv1 ---- */
    memset(c->dx, 0, (size_t)P * nn * sizeof(float));
    conv3x3_bwd(net->params + net->off_c1w, c->z1, c->d_h1, c->x, P, C, net->size,
                g + net->off_c1w, g + net->off_c1b, c->dx);

    return (float)(ce + vl + (double)own_weight * ol);
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
} NetHeader;          /* 旧格式（无 blocks）*/

typedef struct {
    uint32_t magic;
    int32_t  size, planes, channels, vhidden, flags, n_params;
} NetHeader2;         /* 新格式：flags 低 8 位是特性位，高位是残差块数 */
/* NetHeader2 的第 5 个 int32 在老文件里叫 blocks。新代码按 flags 解释，
   并用 n_params 交叉核对（见 net_decode_flags），所以两种文件都能读。 */

/* 解析头部字段 -> (blocks, own_head, vdist)。
   规则：
     1) bit0 置位 = 新布局：blocks = flags >> 8，bit1/bit2 是额外头；
     2) bit0 清零 = 老文件，整个字段就是 blocks（无额外头）；
     3) 新解释算出来的 n_params 与头部不符，而老解释符合 -> 按老解释
        （老文件 blocks 恰好是奇数时会命中这条，保证向后兼容）。 */
static void net_decode_flags(int32_t raw, int nn, int planes, int channels, int vhidden,
                             int n_params, int *blocks, int *own_head, int *vdist) {
    const uint32_t f = (uint32_t)raw;
    int b = (int)(f >> 8);
    int o = (f & NET_FLAG_OWN) ? 1 : 0;
    int v = (f & NET_FLAG_VDIST) ? 1 : 0;
    if (!(f & NET_FLAG_BLOCKS)) {
        b = (int)f; o = 0; v = 0;
    } else if (params_for(nn, planes, channels, vhidden, b, o, v) != n_params) {
        const int b_legacy = (int)f;
        if (params_for(nn, planes, channels, vhidden, b_legacy, 0, 0) == n_params) {
            b = b_legacy; o = 0; v = 0;
        }
    }
    *blocks = b; *own_head = o; *vdist = v;
}

bool net_save(const Net *net, const char *path) {
    /* 先写临时文件再改名：这样正在对弈的程序读到的永远是完整文件
       （训练器每轮覆写 latest.bin，若直接写会有一瞬间是半截文件） */
    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.tmp%ld", path, goai_pid());
    FILE *f = fopen(tmp, "wb");
    if (!f) return false;
    NetHeader2 h;
    h.magic = NET_MAGIC2;
    h.size = net->size; h.planes = net->planes; h.channels = net->channels;
    h.vhidden = net->vhidden;
    h.flags = (int32_t)(NET_FLAG_BLOCKS | (net->own_head ? NET_FLAG_OWN : 0) |
                        (net->vdist ? NET_FLAG_VDIST : 0) | ((uint32_t)net->blocks << 8));
    h.n_params = net->n_params;
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
    int blocks = 0, own = 0, vdist = 0;
    size_t hdr = sizeof(NetHeader);
    if (h.magic == NET_MAGIC2) {
        if (len < sizeof(NetHeader2)) return false;
        NetHeader2 h2;
        memcpy(&h2, data, sizeof(h2));
        h.size = h2.size; h.planes = h2.planes; h.channels = h2.channels;
        h.vhidden = h2.vhidden; h.n_params = h2.n_params;
        net_decode_flags(h2.flags, h.size * h.size, h.planes, h.channels, h.vhidden,
                         h.n_params, &blocks, &own, &vdist);
        hdr = sizeof(NetHeader2);
    } else if (h.magic != NET_MAGIC) {
        return false;
    }
    if (h.n_params <= 0 || len < hdr + (size_t)h.n_params * sizeof(float)) {
        fprintf(stderr, "net_load_mem: 内置权重长度不对（%zu 字节，需要 %zu）\n",
                len, hdr + (size_t)h.n_params * sizeof(float));
        return false;
    }
    if (net->params == NULL || net->size != h.size || net->planes != h.planes ||
        net->channels != h.channels || net->vhidden != h.vhidden || net->n_params != h.n_params ||
        net->blocks != blocks || net->own_head != own || net->vdist != vdist) {
        net_free(net);
        net_init_full(net, h.size, h.planes, h.channels, h.vhidden, blocks, own, vdist, 1);
    }
    memcpy(net->params, (const unsigned char *)data + hdr,
           (size_t)net->n_params * sizeof(float));
    return true;
}

bool net_load(Net *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1) { fclose(f); return false; }
    NetHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = magic;
    int blocks = 0, own = 0, vdist = 0;
    if (magic == NET_MAGIC2) {
        NetHeader2 h2;
        memset(&h2, 0, sizeof(h2));
        h2.magic = magic;
        if (fread((unsigned char *)&h2 + sizeof(magic), sizeof(h2) - sizeof(magic), 1, f) != 1) {
            fclose(f); return false;
        }
        h.size = h2.size; h.planes = h2.planes; h.channels = h2.channels;
        h.vhidden = h2.vhidden; h.n_params = h2.n_params;
        net_decode_flags(h2.flags, h.size * h.size, h.planes, h.channels, h.vhidden,
                         h.n_params, &blocks, &own, &vdist);
    } else if (magic == NET_MAGIC) {
        if (fread((unsigned char *)&h + sizeof(magic), sizeof(h) - sizeof(magic), 1, f) != 1) {
            fclose(f); return false;
        }
    } else {
        fclose(f); return false;
    }
    if (net->params == NULL || net->size != h.size || net->planes != h.planes ||
        net->channels != h.channels || net->vhidden != h.vhidden || net->n_params != h.n_params ||
        net->blocks != blocks || net->own_head != own || net->vdist != vdist) {
        net_free(net);
        net_init_full(net, h.size, h.planes, h.channels, h.vhidden, blocks, own, vdist, 1);
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
