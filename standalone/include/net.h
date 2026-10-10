/* net.h - tiny AlphaZero-style policy/value convolutional network (pure C) */
#ifndef GOAI_NET_H
#define GOAI_NET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "board.h"

#define NET_FEATURE_PLANES 4
#define NET_MAGIC 0x474F4149u  /* "GOAI"  旧格式：无 blocks 字段 */
#define NET_MAGIC2 0x474F414Au /* "GOAJ"  带 flags 字段（低 8 位特性位，高位 blocks） */

/* ---- GOAJ 头部 flags 字段（第 ⑤ 步）------------------------------------
   bit0     : 新布局标记（本头部用 flags 语义，blocks 在 flags >> 8）
   bit1     : 有领地（ownership）辅助头
   bit2     : 值头输出 NET_VALUE_BUCKETS 个桶的分布（否则 1 个标量 + tanh）
   低 8 位留给特性位，blocks 放高位。

   向后兼容：老文件（GOAJ）把这个字段直接当 blocks 写。blocks 为偶数时低 8 位
   恰好是 0，读出来就是"无额外头"；blocks 为奇数时会误判，所以加载时再用
   n_params 交叉核对，对不上就退回"整个字段 = blocks"的旧解释（见 net_decode_flags）。
   GOAI（第一个 magic）没有这个字段，一律按旧布局读。                          */
#define NET_FLAG_BLOCKS 0x01u
#define NET_FLAG_OWN    0x02u
#define NET_FLAG_VDIST  0x04u
#define NET_FLAG_BITS   0xFFu

/* 值目标分布化的桶数：33 个桶均匀覆盖 [-1, +1]，桶心 = -1 + 2k/(K-1) */
#define NET_VALUE_BUCKETS 33

typedef struct {
    int   size;      /* board size              */
    int   nn;        /* size * size             */
    int   planes;    /* input feature planes    */
    int   channels;  /* conv channels           */
    int   vhidden;   /* value head hidden width */
    int   n_params;
    int   blocks;    /* 残差块数量（0 = 原有的两层卷积）   */
    uint32_t flags;  /* NET_FLAG_* + (blocks << 8)，即文件的头部字段 */
    int   own_head;  /* 1 = 有领地辅助头                     */
    int   vdist;     /* 1 = 值头是 K 桶分布（否则标量 tanh）  */
    int   nbuckets;  /* vdist 时 = NET_VALUE_BUCKETS，否则 1  */
    int   off_c1w, off_c1b, off_c2w, off_c2b;
    int   off_pw, off_pb;
    int   off_pfcw, off_pfcb;
    int   off_vw, off_vb;
    int   off_vfc1w, off_vfc1b, off_vfc2w, off_vfc2b;
    int   off_ow, off_ob;                /* 领地头 1x1 conv C->1 */
    int   len_c1w, len_c1b, len_c2w, len_c2b;
    int   len_pw, len_pb, len_pfcw, len_pfcb;
    int   len_vw, len_vb, len_vfc1w, len_vfc1b, len_vfc2w, len_vfc2b;
    int   len_ow, len_ob;
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
    float *zo;                     /* nn 领地 logits（sigmoid 前的值） */
    float *za, *ha;                /* H                          */
    float *vlogits, *vprobs;       /* NET_VALUE_BUCKETS（值分布） */
    float *dx;                     /* P * nn  (grad wrt input)   */
    float *d_h2, *d_pre2, *d_h1, *d_pre1;  /* C * nn scratch     */
    float *d_zp;                   /* 2 * nn scratch             */
    float *d_zv;                   /* nn scratch                 */
    float *d_zo;                   /* nn 领地梯度暂存             */
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
/* 完整版本：额外控制领地头 / 值分布头（own_head、vdist 为 0 时与 net_init_ex 一致） */
void  net_init_full(Net *net, int size, int planes, int channels, int vhidden,
                    int blocks, int own_head, int vdist, uint64_t seed);
void  net_free(Net *net);
int   net_param_count(const Net *net);
float net_param_norm(const Net *net);
void  net_zero_grad(const Net *net, float *grad);
/* 结构描述，例如 "32ch 2blk own vdist"（用于日志） */
void  net_arch_str(const Net *net, char *buf, size_t n);

/* 值分布：第 k 个桶的中心 / z 落入的桶号（z 会被夹到 [-1,1]） */
float net_value_bucket_center(int k);
int   net_value_bucket_index(float z);

/* 把 src 中形状相同的参数段拷贝到 dst（给已有网络加残差块/新头时用）*/
void  net_copy_shared(Net *dst, const Net *src);

void  net_cache_init(const Net *net, NetCache *c);
void  net_cache_free(NetCache *c);

/* forward pass (cache is required: it also holds the activations for backprop) */
void  net_forward(const Net *net, NetCache *c, const float *x, float *policy, float *value);

/* accumulate gradients of loss = CE(policy, pi) + (value - z)^2 ; returns total loss */
float net_backward(const Net *net, NetCache *c, const float *pi, float z, float *grad,
                   float *out_policy_loss, float *out_value_loss);

/* 完整目标：策略 + 价值 + OWN_WEIGHT * 领地 BCE。
   own_target 是 nn 个 [0,1] 的目标（当前行棋方视角，NULL = 这次不算领地损失）；
   返回总损失（含 own_weight 加权），out_* 分别给出未加权的 CE / 价值损失 / 平均 BCE。 */
float net_backward_ex(const Net *net, NetCache *c, const float *pi, float z,
                      const float *own_target, float own_weight, float *grad,
                      float *out_policy_loss, float *out_value_loss, float *out_own_loss);

void  net_adam(Net *net, float *grad, float *m, float *v, int step, float lr,
               float weight_decay, float clip_norm);

bool  net_save(const Net *net, const char *path);
bool  net_load(Net *net, const char *path);
/* 从内存加载权重（内置权重用） */
bool  net_load_mem(Net *net, const void *data, size_t len);

/* encode the position from the point of view of the player to move */
void  net_features(const Board *b, float *x);

#endif /* GOAI_NET_H */
