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
    int    sims;             /* MCTS simulations per move in self-play   */
    int    games_per_iter;
    int    iterations;
    int    train_steps;      /* gradient steps per iteration             */
    int    batch_size;
    float  lr;
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
    int    forever;          /* keep iterating until stopped        */
    int    resume;           /* load --resume weights first         */
    const char *resume_path;
    const char *status_file; /* live status file for monitors       */
    const char *stop_file;   /* create this file to stop gracefully */
    int    gate;             /* keep best.bin only if it improves   */
    int    gate_games;       /* games per promotion check           */
    int    eval_every;       /* promotion check every N iterations  */
    int    plot;             /* refresh the curve PNG every round   */
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
    volatile int games_done;   /* self-play games finished (any thread) */
    int      xlen;           /* planes * nn (bytes, features are 0/1)    */
    int      pilen;          /* nn + 1                                   */
    uint8_t *bx;
    float   *bpi;
    float   *bz;
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
void  buffer_push(Trainer *t, const uint8_t *x, const float *pi, float z);
int   train_gradient_steps(Trainer *t, const TrainConfig *cfg, int steps,
                           float *policy_loss, float *value_loss);

int   selfplay_game(const TrainConfig *cfg, Search *s, Rng *rng, Sgf *sgf, ExampleBatch *out);
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
