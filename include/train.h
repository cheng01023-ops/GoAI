/* train.h - self-play reinforcement learning (simplified AlphaZero) */
#ifndef GOAI_TRAIN_H
#define GOAI_TRAIN_H

#include <pthread.h>

#include "mcts.h"
#include "net.h"
#include "rand.h"

typedef struct {
    int    size;
    int    planes, channels, vhidden;
    int    blocks;            /* 残差块数量（0 = 原来的两层卷积）*/
    int    sims;             /* MCTS simulations per move in self-play   */
    int    games_per_iter;
    int    iterations;
    int    train_steps;      /* gradient steps per iteration             */
    int    batch_size;
    float  lr;
    float  lr_min;           /* 学习率下限（衰减不会低于它）          */
    int    lr_decay_every;   /* 每多少轮衰减一次（0 = 不衰减）        */
    float  lr_decay_factor;  /* 每次衰减乘的系数（例如 0.7）          */
    float  weight_decay;
    float  dirichlet_alpha;
    float  noise_eps;
    int    temp_moves;       /* plies played with temperature 1          */
    double komi;
    int    buffer_cap;       /* replay buffer capacity (positions)       */
    int    eval_games;
    int    eval_sims;
    int    max_moves;        /* game length cap                          */
    int    pass_min_move;
    int    threads;          /* self-play threads (0 = auto)        */
    int    augment;          /* 8-fold symmetry data augmentation   */
    /* ---- 第 ⑤ 步：辅助目标 ---- */
    float  own_weight;       /* 领地 BCE 权重（>0 时网络带领地头，0 = 关闭）*/
    int    vdist;            /* 1 = 值头改成 NET_VALUE_BUCKETS 桶分布      */
    int    forever;          /* keep iterating until stopped        */
    int    resume;           /* load --resume weights first         */
    const char *resume_path;
    const char *status_file; /* live status file for monitors       */
    const char *stop_file;   /* create this file to stop gracefully */
    int    gate;             /* keep best.bin only if it improves   */
    int    gate_games;       /* games per promotion check           */
    int    eval_every;       /* promotion check every N iterations  */
    int    plot;             /* refresh the curve PNG every round   */
    /* ---- 固定锚点评估：对同一个基准网络定期对战，得到低噪声的进步曲线 ---- */
    const char *anchor_path;  /* 基准网络 .bin（NULL = 关闭）          */
    int    anchor_every;      /* 每多少轮评估一次（0 = 关闭）          */
    int    anchor_games;      /* 每次评估多少局                        */
    /* ---- 退化保护：连续多次晋级赛表现差就回滚到 best.bin ---- */
    int    open_plies;        /* 成对开局：每对先随机下这么多手（0=关闭）*/
    uint64_t open_seed;       /* 开局生成的种子（固定值 => 各实验可比）   */
    int    reuse;             /* 1 = 自对弈复用搜索树（默认关，见 README）*/
    int    rollback;          /* 1 = 打开回滚保护                      */
    int    rollback_patience; /* 连续几次差就回滚                      */
    int    save_sgf;         /* how many self-play games to store as SGF */
    uint64_t seed;
    const char *out_dir;
} TrainConfig;

/* one self-play game worth of training examples */
typedef struct {
    uint8_t *X;
    float   *PI;
    float   *Z;
    int      n, cap, xlen, pilen;
} ExampleBatch;

typedef struct {
    Net      net;
    NetCache cache;
    float   *grad, *adam_m, *adam_v;
    int      step;
    float    lr;          /* 当前学习率（每轮按计划更新） */
    volatile int games_done;   /* self-play games finished (any thread) */
    int      xlen;           /* planes * nn (bytes, features are 0/1)    */
    int      pilen;          /* nn + 1                                   */
    int      nn;             /* size * size                              */
    uint8_t *bx;
    float   *bpi;
    float   *bz;
    int32_t *bown;           /* 每个样本所属棋局号（-1 = 无归属标签）      */
    uint8_t *bwhite;         /* 每个样本是否轮到白棋                     */
    /* 终局归属只按"一盘棋一份"存（81 字节/q，省内存），样本里只记棋局号；
       棋局号取模映射到槽位，槽位被新棋局覆盖后旧样本会自动退回"无标签"。 */
    int8_t  *own_tab;        /* own_slots * nn，黑方视角归属             */
    int32_t *own_gid;        /* 槽位对应棋局号（-1 = 空）                 */
    int      own_slots;
    int32_t  next_gid;
    long     own_miss;       /* 采样时标签已失效的样本数（统计用）          */
    int      cap, count, head;
    Rng      rng;
    pthread_mutex_t buf_lock;
} Trainer;

typedef struct {
    const Net *net;       /* NULL -> random mover                        */
    int        sims;      /* MCTS simulations (0 with net == NULL)       */
    int        random_mover;
} EngineSpec;

typedef struct Sgf Sgf;

double goai_now(void);
void  train_config_default(TrainConfig *cfg, int size);
int   trainer_init(Trainer *t, const TrainConfig *cfg, uint64_t seed);
void  trainer_free(Trainer *t);
void  buffer_push(Trainer *t, const uint8_t *x, const float *pi, float z, int gid, int white);
/* 提交一盘棋：登记终局归属（own_black = nn 个黑方视角 {-1,0,+1}）+ 推入所有样本 */
void  trainer_push_game(Trainer *t, const ExampleBatch *batch, const int8_t *own_black);
/* 取某盘棋的终局归属表（黑方视角）；已被覆盖时返回 NULL */
const int8_t *trainer_own_labels(const Trainer *t, int gid);
int   train_gradient_steps(Trainer *t, const TrainConfig *cfg, int steps,
                           float *policy_loss, float *value_loss, float *own_loss);
/* 学习率计划：绝对轮次每 lr_decay_every 轮乘一次 lr_decay_factor，不低于 lr_min */
float train_lr_at_iter(const TrainConfig *cfg, int iter);
/* 胜率的 95% 置信区间半宽（用于判断"进步是否在噪声内"） */
double train_winrate_ci(double p, int games);

/* own_out（可空）回填终局归属（nn 个，黑方视角 {-1,0,+1}）*/
int   selfplay_game(const TrainConfig *cfg, Search *s, Rng *rng, Sgf *sgf, ExampleBatch *out,
                    int8_t *own_out);
int   batch_init(ExampleBatch *b, int cap, int xlen, int pilen);
void  batch_free(ExampleBatch *b);
int   eval_match(Search *s, const TrainConfig *cfg, EngineSpec a, EngineSpec b,
                 int games, int *wins_a, int *wins_b, int *draws);

Sgf  *sgf_open(const char *path, int size, double komi, const char *comment);
void  sgf_move(Sgf *g, int move, int color);
void  sgf_result(Sgf *g, int winner);
void  sgf_close(Sgf *g);

int   train_run(const TrainConfig *cfg);
int   train_bench(const TrainConfig *cfg);

#endif /* GOAI_TRAIN_H */
