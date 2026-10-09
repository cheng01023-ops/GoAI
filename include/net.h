/* net.h - tiny AlphaZero-style policy/value convolutional network (pure C) */
#ifndef GOAI_NET_H
#define GOAI_NET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "board.h"

#define NET_FEATURE_PLANES 4
#define NET_MAGIC 0x474F4149u  /* "GOAI"  旧格式：无残差块 */
#define NET_MAGIC2 0x474F414Au /* "GOAJ"  新格式：带 blocks 字段 */

typedef struct {
    int   size;      /* board size              */
    int   nn;        /* size * size             */
    int   planes;    /* input feature planes    */
    int   channels;  /* conv channels           */
    int   vhidden;   /* value head hidden width */
    int   n_params;
    int   blocks;    /* 残差块数量（0 = 原有的两层卷积）   */
    int   off_c1w, off_c1b, off_c2w, off_c2b;
    int   off_pw, off_pb;
    int   off_pfcw, off_pfcb;
    int   off_vw, off_vb;
    int   off_vfc1w, off_vfc1b, off_vfc2w, off_vfc2b;
    int   len_c1w, len_c1b, len_c2w, len_c2b;
    int   len_pw, len_pb, len_pfcw, len_pfcb;
    int   len_vw, len_vb, len_vfc1w, len_vfc1b, len_vfc2w, len_vfc2b;
    int   off_bw, len_bw, blk_stride;   /* 残差块权重区 */
    float *params;
} Net;

typedef struct {
    float *x;                      /* P * nn                     */
    float *z1, *h1;                /* C * nn                     */
    float *z2, *h2;
    float *zp;                     /* 2 * nn                     */
    float *logits, *probs;         /* nn + 1                     */
    float *zv;                     /* nn                         */
    float *za, *ha;                /* H                          */
    float *dx;                     /* P * nn  (grad wrt input)   */
    float *d_h2, *d_pre2, *d_h1, *d_pre1;  /* C * nn scratch     */
    float *d_zp;                   /* 2 * nn scratch             */
    float *d_zv;                   /* nn scratch                 */
    float *d_ha;                   /* H scratch                  */
    float *g_scratch;              /* n_params, used when grad == NULL */
    float *blk;                    /* blocks * 4 * C * nn 前向缓存    */
    float *dblk;                   /* 3 * C * nn 反向暂存             */
    float  vo, v;
} NetCache;

void  net_config_default(int size, int *planes, int *channels, int *vhidden);
void  net_init(Net *net, int size, int planes, int channels, int vhidden, uint64_t seed);
/* 带残差块的版本：blocks = 0 时与 net_init 完全一致 */
void  net_init_ex(Net *net, int size, int planes, int channels, int vhidden,
                  int blocks, uint64_t seed);
void  net_free(Net *net);
int   net_param_count(const Net *net);
float net_param_norm(const Net *net);
void  net_zero_grad(const Net *net, float *grad);

void  net_cache_init(const Net *net, NetCache *c);
void  net_cache_free(NetCache *c);

/* forward pass (cache is required: it also holds the activations for backprop) */
void  net_forward(const Net *net, NetCache *c, const float *x, float *policy, float *value);

/* accumulate gradients of loss = CE(policy, pi) + (value - z)^2 ; returns total loss */
float net_backward(const Net *net, NetCache *c, const float *pi, float z, float *grad,
                   float *out_policy_loss, float *out_value_loss);

void  net_adam(Net *net, float *grad, float *m, float *v, int step, float lr,
               float weight_decay, float clip_norm);

bool  net_save(const Net *net, const char *path);
bool  net_load(Net *net, const char *path);
/* 从内存加载权重（内置权重用） */
bool  net_load_mem(Net *net, const void *data, size_t len);

/* encode the position from the point of view of the player to move */
void  net_features(const Board *b, float *x);

#endif /* GOAI_NET_H */
