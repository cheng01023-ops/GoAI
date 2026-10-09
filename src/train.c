/* train.c - self-play, training loop, evaluation and SGF output */
#include "train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

#include "compat.h"

/* ------------------------------------------------------- runtime helpers */

static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

/* 跨平台优雅停止：出现 STOP 文件（或收到 SIGINT/SIGTERM）即停止 */
static int stop_requested(const TrainConfig *cfg) {
    return g_stop || (cfg && cfg->stop_file && goai_file_exists(cfg->stop_file));
}

int train_stop_requested(void) { return g_stop != 0; }

static void iso_time(char *buf, size_t n) {
    goai_localtime_str(buf, n, "%Y-%m-%d %H:%M:%S");
}

/* Live status file: key=value lines that any monitor (or the launcher) can read. */
static void write_status(const TrainConfig *cfg, const char *state, int iter, int game,
                         double elapsed_iter, double elapsed_total, long total_games,
                         const char *extra_key, double extra_val) {
    char path[1200];
    const char *target = cfg->status_file;
    if (!target) { snprintf(path, sizeof(path), "%s/status.txt", cfg->out_dir); target = path; }
    FILE *f = fopen(target, "w");
    if (!f) return;
    char ts[64];
    iso_time(ts, sizeof(ts));
    fprintf(f, "state=%s\n", state);
    fprintf(f, "iteration=%d\n", iter);
    fprintf(f, "game=%d\n", game);
    fprintf(f, "games_this_iter=%d\n", cfg->games_per_iter);
    fprintf(f, "total_games=%ld\n", total_games);
    fprintf(f, "elapsed_iter=%.1f\n", elapsed_iter);
    fprintf(f, "elapsed_total=%.1f\n", elapsed_total);
    fprintf(f, "threads=%d\n", cfg->threads);
    fprintf(f, "sims=%d\n", cfg->sims);
    fprintf(f, "channels=%d\n", cfg->channels);
    fprintf(f, "pid=%ld\n", goai_pid());
    if (extra_key) fprintf(f, "%s=%.4f\n", extra_key, extra_val);
    fprintf(f, "updated=%s\n", ts);
    fclose(f);
}

typedef struct {
    Trainer           *t;
    const TrainConfig *cfg;
    int                iter;
    int                games_target;
    double             t_iter0;
    double             t_run0;
    long               games_before;
    volatile int       stop;
} ProgressCtx;

static void *progress_thread(void *arg) {
    ProgressCtx *pc = (ProgressCtx *)arg;
    const int tty = isatty(1);
    while (!pc->stop) {
        goai_sleep_ms(2000);
        if (pc->stop) break;
        const int done = __atomic_load_n(&pc->t->games_done, __ATOMIC_RELAXED);
        const double el = goai_now() - pc->t_iter0;
        write_status(pc->cfg, "running", pc->iter, done, el, goai_now() - pc->t_run0,
                     pc->games_before + done, NULL, 0.0);
        if (tty) {
            printf("\r  第 %d 轮 · 自对弈 %d/%d 局 · 本轮 %5.1fs · 累计 %ld 局        ",
                   pc->iter, done, pc->games_target, el, pc->games_before + done);
            fflush(stdout);
        }
    }
    if (tty) { printf("\r%*s\r", 72, ""); fflush(stdout); }
    return NULL;
}


void train_config_default(TrainConfig *cfg, int size) {
    memset(cfg, 0, sizeof(*cfg));
    int planes, channels, vhidden;
    net_config_default(size, &planes, &channels, &vhidden);
    cfg->size = size;
    cfg->planes = planes;
    cfg->channels = channels;
    cfg->vhidden = vhidden;
    cfg->sims = 120;
    cfg->games_per_iter = 30;
    cfg->iterations = 6;
    cfg->train_steps = 300;
    cfg->batch_size = 64;
    cfg->lr = 0.02f;
    cfg->lr_min = 1e-4f;
    cfg->anchor_path = NULL;
    cfg->anchor_every = 20;      /* 每 20 轮对固定基准评估一次 */
    cfg->anchor_games = 20;
    cfg->blocks = 0;
    cfg->open_plies = 4;         /* 成对开局：4 手（大幅降低评估方差） */
    cfg->open_seed = 0xA11CE5EEDULL;
    cfg->reuse = 0;   /* 实测：默认关闭（见 README 的 A/B 结论） */
    cfg->rollback = 1;
    cfg->rollback_patience = 3;
    cfg->lr_decay_every = 0;      /* 默认不衰减，用 --lr-decay-every 打开 */
    cfg->lr_decay_factor = 0.7f;
    cfg->weight_decay = 1e-4f;
    cfg->dirichlet_alpha = 0.3f;
    cfg->noise_eps = 0.25f;
    cfg->temp_moves = 12;
    cfg->komi = 7.0;
    cfg->buffer_cap = 30000;
    cfg->eval_games = 20;
    cfg->eval_sims = 40;
    cfg->max_moves = 3 * size * size;
    cfg->pass_min_move = 2 * size;
    cfg->save_sgf = 3;
    cfg->forever = 0;
    cfg->resume = 0;
    cfg->resume_path = NULL;
    cfg->status_file = NULL;
    cfg->stop_file = NULL;
    cfg->gate = 1;
    cfg->gate_games = 20;
    cfg->eval_every = 3;
    cfg->plot = 1;
    cfg->augment = 1;
    cfg->threads = 0;   /* auto: one per CPU, capped at 8 */
    cfg->seed = 20241001ULL;
    cfg->out_dir = "runs";
}

int trainer_init(Trainer *t, const TrainConfig *cfg, uint64_t seed) {
    memset(t, 0, sizeof(*t));
    t->lr = cfg->lr;
    net_init_ex(&t->net, cfg->size, cfg->planes, cfg->channels, cfg->vhidden, cfg->blocks, seed);
    net_cache_init(&t->net, &t->cache);
    const size_t np = (size_t)t->net.n_params;
    t->xlen = cfg->planes * cfg->size * cfg->size;
    t->pilen = cfg->size * cfg->size + 1;
    t->cap = cfg->buffer_cap;
    t->grad = (float *)calloc(np, sizeof(float));
    t->adam_m = (float *)calloc(np, sizeof(float));
    t->adam_v = (float *)calloc(np, sizeof(float));
    t->bx = (uint8_t *)malloc((size_t)t->cap * (size_t)t->xlen);
    t->bpi = (float *)malloc((size_t)t->cap * (size_t)t->pilen * sizeof(float));
    t->bz = (float *)malloc((size_t)t->cap * sizeof(float));
    if (!t->grad || !t->adam_m || !t->adam_v || !t->bx || !t->bpi || !t->bz) {
        fprintf(stderr, "trainer_init: out of memory\n");
        return 1;
    }
    rng_seed(&t->rng, seed ^ 0x5DEECE66DULL);
    pthread_mutex_init(&t->buf_lock, NULL);
    return 0;
}

void trainer_free(Trainer *t) {
    net_cache_free(&t->cache);
    net_free(&t->net);
    free(t->grad); free(t->adam_m); free(t->adam_v);
    free(t->bx); free(t->bpi); free(t->bz);
    pthread_mutex_destroy(&t->buf_lock);
    memset(t, 0, sizeof(*t));
}

void buffer_push(Trainer *t, const uint8_t *x, const float *pi, float z) {
    const int idx = t->head;
    memcpy(t->bx + (size_t)idx * t->xlen, x, (size_t)t->xlen);
    memcpy(t->bpi + (size_t)idx * t->pilen, pi, (size_t)t->pilen * sizeof(float));
    t->bz[idx] = z;
    t->head = (idx + 1) % t->cap;
    if (t->count < t->cap) t->count++;
}

/* ------------------------------------------------------------------ SGF */

struct Sgf { FILE *f; int size; int n; };

Sgf *sgf_open(const char *path, int size, double komi, const char *comment) {
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    Sgf *g = (Sgf *)calloc(1, sizeof(Sgf));
    if (!g) { fclose(f); return NULL; }
    g->f = f;
    g->size = size;
    fprintf(f, "(;GM[1]FF[4]CA[UTF-8]AP[GoAI:1.0]SZ[%d]KM[%.1f]PB[Black]PW[White]"
               "RU[Chinese]C[%s]\n", size, komi, comment ? comment : "self-play");
    return g;
}

void sgf_move(Sgf *g, int move, int color) {
    if (!g) return;
    if (move == M_PASS) fprintf(g->f, ";%c[]", color == 1 ? 'B' : 'W');
    else fprintf(g->f, ";%c[%c%c]", color == 1 ? 'B' : 'W',
                 'a' + (move % g->size), 'a' + (move / g->size));
    if (++g->n % 10 == 0) fprintf(g->f, "\n");
}

void sgf_result(Sgf *g, int winner) {
    if (!g) return;
    if (winner == 0) fprintf(g->f, "RE[0]");
    else fprintf(g->f, "RE[%c+R]", winner == 1 ? 'B' : 'W');
}

void sgf_close(Sgf *g) {
    if (!g) return;
    fprintf(g->f, ")\n");
    fclose(g->f);
    free(g);
}

/* ------------------------------------------------------------- self play */

/* the eight symmetries of the square, used for data augmentation */
static void transform_xy(int t, int n, int x, int y, int *ox, int *oy) {
    switch (t & 7) {
        case 0: *ox = x;         *oy = y;         break;
        case 1: *ox = n - 1 - y; *oy = x;         break;
        case 2: *ox = n - 1 - x; *oy = n - 1 - y; break;
        case 3: *ox = y;         *oy = n - 1 - x; break;
        case 4: *ox = n - 1 - x; *oy = y;         break;
        case 5: *ox = x;         *oy = n - 1 - y; break;
        case 6: *ox = n - 1 - y; *oy = n - 1 - x; break;
        default:*ox = y;         *oy = x;         break;
    }
}

int batch_init(ExampleBatch *b, int cap, int xlen, int pilen) {
    memset(b, 0, sizeof(*b));
    b->cap = cap; b->xlen = xlen; b->pilen = pilen;
    b->X  = (uint8_t *)malloc((size_t)cap * (size_t)xlen);
    b->PI = (float *)malloc((size_t)cap * (size_t)pilen * sizeof(float));
    b->Z  = (float *)malloc((size_t)cap * sizeof(float));
    if (!b->X || !b->PI || !b->Z) { free(b->X); free(b->PI); free(b->Z); return 1; }
    return 0;
}

void batch_free(ExampleBatch *b) {
    free(b->X); free(b->PI); free(b->Z);
    memset(b, 0, sizeof(*b));
}

/* play one self-play game and fill `out` with (features, policy target, z) */
int selfplay_game(const TrainConfig *cfg, Search *s, Rng *rng, Sgf *sgf, ExampleBatch *out) {
    (void)rng;
    out->n = 0;
    float xbuf[BOARD_MAX_POINTS * 8];
    float pol[BOARD_MAX_POINTS + 1];
    Board b;
    board_init(&b, cfg->size);
    while (b.passes < 2 && b.nmoves < cfg->max_moves && out->n < out->cap) {
        const float temp = (out->n < cfg->temp_moves) ? 1.0f : 0.0f;
        const int color = b.to_move;
        const int move = search_run(s, &b, cfg->sims, cfg->dirichlet_alpha, cfg->noise_eps,
                                    temp, pol, NULL);
        net_features(&b, xbuf);
        uint8_t *dst = out->X + (size_t)out->n * out->xlen;
        for (int i = 0; i < out->xlen; i++) dst[i] = xbuf[i] > 0.5f ? 1 : 0;
        memcpy(out->PI + (size_t)out->n * out->pilen, pol, (size_t)out->pilen * sizeof(float));
        if (sgf) sgf_move(sgf, move, color);
        out->n++;
        if (!board_play(&b, move)) board_play(&b, M_PASS);
        s->pending_advance = cfg->reuse ? move : -1;   /* 树复用开关 */
    }
    const int winner = board_winner(&b, cfg->komi);
    if (sgf) sgf_result(sgf, winner);
    for (int i = 0; i < out->n; i++) {
        const int player = (i % 2 == 0) ? 1 : 2;
        out->Z[i] = (winner == 0) ? 0.0f : (winner == player ? 1.0f : -1.0f);
    }
    return out->n;
}

/* ------------------------------------------------- parallel self-play --- */

typedef struct {
    const TrainConfig *cfg;
    Trainer           *t;
    Search             search;
    Rng                rng;
    int               *game_ids;
    int                ngames;
    int                iter;
    int                positions;
} SelfPlayJob;

static void *selfplay_thread(void *arg) {
    SelfPlayJob *j = (SelfPlayJob *)arg;
    const TrainConfig *cfg = j->cfg;
    Trainer *t = j->t;
    ExampleBatch batch;
    if (batch_init(&batch, cfg->max_moves + 2, t->xlen, t->pilen) != 0) return NULL;
    for (int k = 0; k < j->ngames; k++) {
        const int gid = j->game_ids[k];
        Sgf *sgf = NULL;
        if (gid < cfg->save_sgf) {
            char path[1024], cmt[160];
            snprintf(path, sizeof(path), "%s/games/iter%02d_game%02d.sgf", cfg->out_dir, j->iter, gid + 1);
            snprintf(cmt, sizeof(cmt), "GoAI self-play iteration %d game %d", j->iter, gid + 1);
            sgf = sgf_open(path, cfg->size, cfg->komi, cmt);
        }
        const int n = selfplay_game(cfg, &j->search, &j->rng, sgf, &batch);
        if (sgf) sgf_close(sgf);
        if (n > 0) {
            pthread_mutex_lock(&t->buf_lock);
            for (int i = 0; i < batch.n; i++)
                buffer_push(t, batch.X + (size_t)i * t->xlen,
                               batch.PI + (size_t)i * t->pilen, batch.Z[i]);
            pthread_mutex_unlock(&t->buf_lock);
            j->positions += batch.n;
        }
        __atomic_add_fetch(&t->games_done, 1, __ATOMIC_RELAXED);
        if (g_stop || goai_file_exists(j->cfg->stop_file)) break;
    }
    batch_free(&batch);
    return NULL;
}

/* --------------------------------------------------------------- training */

/* 胜率的 95% 置信区间半宽：1.96 * sqrt(p(1-p)/n)。
   用来判断"这次进步"是真的还是只是样本噪声。 */
double train_winrate_ci(double p, int games) {
    if (games <= 0) return 0.0;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return 1.96 * sqrt(p * (1.0 - p) / (double)games);
}

/* 学习率计划：按"绝对轮次"衰减，所以中途停止再 --resume 接着练不会打乱计划。
   第 k 轮的学习率 = lr * factor^(k / every)，并夹在 [lr_min, lr] 之间。 */
float train_lr_at_iter(const TrainConfig *cfg, int iter) {
    if (cfg->lr_decay_every <= 0) return cfg->lr;
    if (!(cfg->lr_decay_factor > 0.0f) || cfg->lr_decay_factor >= 1.0f) return cfg->lr;
    const int k = iter / cfg->lr_decay_every;
    if (k <= 0) return cfg->lr;
    float lr = cfg->lr * powf(cfg->lr_decay_factor, (float)k);
    if (lr < cfg->lr_min) lr = cfg->lr_min;
    return lr;
}

int train_gradient_steps(Trainer *t, const TrainConfig *cfg, int steps,
                         float *policy_loss, float *value_loss) {
    if (t->count == 0) return 1;
    double pl = 0, vl = 0;
    float *x = (float *)malloc((size_t)t->xlen * sizeof(float));
    float *pi = (float *)malloc((size_t)t->pilen * sizeof(float));
    if (!x || !pi) { free(x); free(pi); return 1; }
    for (int s = 0; s < steps; s++) {
        net_zero_grad(&t->net, t->grad);
        double bpl = 0, bvl = 0;
        for (int k = 0; k < cfg->batch_size; k++) {
            const int idx = (int)rng_below(&t->rng, (uint32_t)t->count);
            const uint8_t *src = t->bx + (size_t)idx * t->xlen;
            const float *pi_src = t->bpi + (size_t)idx * t->pilen;
            const int n = cfg->size, nn = n * n, planes = cfg->planes;
            const int tt = (int)rng_below(&t->rng, 8);      /* random symmetry */
            if (cfg->augment) {
                for (int pl = 0; pl < planes; pl++) {
                    for (int y = 0; y < n; y++) for (int xx = 0; xx < n; xx++) {
                        int ax, ay;
                        transform_xy(tt, n, xx, y, &ax, &ay);
                        x[pl * nn + ay * n + ax] = (float)src[pl * nn + y * n + xx];
                    }
                }
                for (int i = 0; i < t->pilen; i++) pi[i] = 0.0f;
                pi[nn] = pi_src[nn];                        /* pass is invariant */
                for (int y = 0; y < n; y++) for (int xx = 0; xx < n; xx++) {
                    int ax, ay;
                    transform_xy(tt, n, xx, y, &ax, &ay);
                    pi[ay * n + ax] = pi_src[y * n + xx];
                }
            } else {
                for (int i = 0; i < t->xlen; i++) x[i] = (float)src[i];
                memcpy(pi, pi_src, (size_t)t->pilen * sizeof(float));
            }
            float l1 = 0, l2 = 0;
            net_forward(&t->net, &t->cache, x, NULL, NULL);
            net_backward(&t->net, &t->cache, pi, t->bz[idx], t->grad, &l1, &l2);
            bpl += l1;
            bvl += l2;
        }
        const float scale = 1.0f / (float)cfg->batch_size;
        for (int i = 0; i < t->net.n_params; i++) t->grad[i] *= scale;
        net_adam(&t->net, t->grad, t->adam_m, t->adam_v, ++t->step,
                 t->lr, cfg->weight_decay, 1.0f);
        pl += bpl * scale;
        vl += bvl * scale;
    }
    free(x); free(pi);
    if (policy_loss) *policy_loss = (float)(pl / steps);
    if (value_loss)  *value_loss  = (float)(vl / steps);
    return 0;
}

/* ------------------------------------------------------------- evaluation */

static int eval_move(Search *s, const TrainConfig *cfg, EngineSpec e, const Board *b, Rng *rng) {
    if (e.random_mover || !e.net) return board_random_move(b, rng, cfg->pass_min_move);
    search_set_net(s, e.net);
    float pol[BOARD_MAX_POINTS + 1];
    const float temp = (b->nmoves < 4) ? 1.0f : 0.0f;   /* a little opening variety */
    return search_run(s, b, e.sims, 0.0f, 0.0f, temp, pol, NULL);
}

/* 生成一个随机开局：从空盘随机下 plies 手合法着法（不停着）。
   用固定种子，所以不同实验（A/B 两臂、不同时间）拿到的是同一批开局，
   这样比较才有意义。 */
static void make_opening(const TrainConfig *cfg, Board *bd, int pair_index) {
    board_init(bd, cfg->size);
    if (cfg->open_plies <= 0) return;
    Rng rng;
    rng_seed(&rng, cfg->open_seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(pair_index + 1));
    int moves[BOARD_MAX_POINTS + 1];
    for (int i = 0; i < cfg->open_plies; i++) {
        const int nm = board_legal_moves(bd, moves, false);
        if (nm <= 0) break;
        if (!board_play(bd, moves[rng_below(&rng, (uint32_t)nm)])) break;
    }
}

/* 一盘对局：从 start 局面开始，a_is_black 决定谁执黑。 */
static int play_one(Search *s, const TrainConfig *cfg, EngineSpec a, EngineSpec b,
                    const Board *start, int a_is_black) {
    Board bd = *start;
    while (bd.passes < 2 && bd.nmoves < cfg->max_moves) {
        const EngineSpec e = (bd.to_move == 1) ? (a_is_black ? a : b) : (a_is_black ? b : a);
        const int mv = eval_move(s, cfg, e, &bd, &s->rng);
        if (!board_play(&bd, mv)) board_play(&bd, M_PASS);
    }
    return board_winner(&bd, cfg->komi);
}

int eval_match(Search *s, const TrainConfig *cfg, EngineSpec a, EngineSpec b,
               int games, int *wins_a, int *wins_b, int *draws) {
    int wa = 0, wb = 0, dr = 0;
    int g = 0;
    /* 成对开局：同一开局下 A 执黑一次、执白一次，抵消开局运气 */
    if (cfg->open_plies > 0) {
        const int pairs = games / 2;
        for (int p = 0; p < pairs; p++) {
            Board open_bd;
            make_opening(cfg, &open_bd, p);
            for (int side = 0; side < 2; side++) {
                const int a_is_black = (side == 0);
                const int w = play_one(s, cfg, a, b, &open_bd, a_is_black);
                if (w == 0) dr++;
                else if ((w == 1) == a_is_black) wa++;
                else wb++;
            }
            g += 2;
        }
    }
    for (; g < games; g++) {   /* 剩余的（或者关闭成对开局时）走原来的逻辑 */
        Board bd;
        board_init(&bd, cfg->size);
        const int a_is_black = (g % 2 == 0);
        const int w = play_one(s, cfg, a, b, &bd, a_is_black);
        if (w == 0) dr++;
        else if ((w == 1) == a_is_black) wa++;
        else wb++;
    }
    if (wins_a) *wins_a = wa;
    if (wins_b) *wins_b = wb;
    if (draws)  *draws  = dr;
    return 0;
}

/* ------------------------------------------------------------ train loop */

static void mkdir_p(const char *path) { (void)goai_mkdir_p(path); }

int train_run(const TrainConfig *cfgin) {
    TrainConfig cfg = *cfgin;
    if (cfg.threads <= 0) {
        long ncpu = goai_cpu_count();
        cfg.threads = (ncpu > 0) ? (int)ncpu : 1;
        if (cfg.threads > 8) cfg.threads = 8;
    }
    if (cfg.eval_every <= 0) cfg.eval_every = 3;
    if (cfg.gate_games <= 0) cfg.gate_games = 20;

    char path[1200];
    char status_path[1200];
    char best_path[1200];
    mkdir_p(cfg.out_dir);
    snprintf(path, sizeof(path), "%s/games", cfg.out_dir);
    mkdir_p(path);
    if (!cfg.status_file) {
        snprintf(status_path, sizeof(status_path), "%s/status.txt", cfg.out_dir);
        cfg.status_file = status_path;
    }
    snprintf(best_path, sizeof(best_path), "%s/best.bin", cfg.out_dir);
    {
        static char stop_buf[1200];
        snprintf(stop_buf, sizeof(stop_buf), "%s/STOP", cfg.out_dir);
        cfg.stop_file = stop_buf;
        remove(cfg.stop_file);          /* 清掉上一次遗留的停止标记 */
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#ifdef SIGHUP
    signal(SIGHUP, on_signal);
#endif

    Trainer t;
    if (trainer_init(&t, &cfg, cfg.seed) != 0) return 1;
    Search s;
    search_init(&s, cfg.size, &t.net, cfg.seed ^ 0xABCDEFULL, cfg.komi);
    s.pass_min_move = cfg.pass_min_move;
    s.max_moves = cfg.max_moves;

    /* ---- resume ---------------------------------------------------------- */
    int  start_iter = 1;
    long total_games = 0;
    if (cfg.resume && cfg.resume_path) {
        if (cfg.blocks > 0) {
            /* 带残差块时不能直接 net_load（那会把网络重建成文件里的结构），
               而是先读进临时网络，再把形状相同的权重搬进已经建好的残差网络。 */
            Net old;
            memset(&old, 0, sizeof(old));
            if (net_load(&old, cfg.resume_path)) {
                if (old.blocks != cfg.blocks) {
                    net_copy_shared(&t.net, &old);
                    printf("继续训练：已载入 %s（残差块 %d -> %d，共享权重已迁移）\n",
                           cfg.resume_path, old.blocks, cfg.blocks);
                } else {
                    memcpy(t.net.params, old.params, (size_t)t.net.n_params * sizeof(float));
                    printf("继续训练：已载入 %s\n", cfg.resume_path);
                }
                net_free(&old);
            } else {
                printf("继续训练：%s 不存在，将从随机初始化开始\n", cfg.resume_path);
            }
        } else if (net_load(&t.net, cfg.resume_path)) {
            printf("继续训练：已载入 %s\n", cfg.resume_path);
        } else {
            printf("继续训练：%s 不存在，将从随机初始化开始\n", cfg.resume_path);
        }
    }
    {
        snprintf(path, sizeof(path), "%s/state.txt", cfg.out_dir);
        FILE *f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%d %ld", &start_iter, &total_games) != 2) { start_iter = 1; total_games = 0; }
            fclose(f);
        }
    }
    if (start_iter < 1) start_iter = 1;

    if (cfg.gate) {
        FILE *bf = fopen(best_path, "rb");
        if (!bf) net_save(&t.net, best_path);
        else fclose(bf);
    }

    /* ---- log ------------------------------------------------------------- */
    int log_has_header = 0;
    snprintf(path, sizeof(path), "%s/train_log.csv", cfg.out_dir);
    {
        FILE *cf = fopen(path, "r");
        if (cf) { log_has_header = 1; fclose(cf); }
    }
    FILE *csv = fopen(path, "a");
    if (csv && !log_has_header) {
        fprintf(csv, "iteration,games,positions,policy_loss,value_loss,param_norm,"
                     "winrate_vs_random,gate_winrate,elapsed_s,lr,anchor_winrate,selfplay_s,train_s,eval_s\n");
        fflush(csv);
    }

    printf("=== GoAI 持续自我迭代训练 ===\n");
    printf("  棋盘 %dx%d · %d 通道 · 每步 %d 次模拟 · 每轮 %d 局 · %d 线程\n",
           cfg.size, cfg.size, cfg.channels, cfg.sims, cfg.games_per_iter, cfg.threads);
    printf("  输出目录 %s/  （latest.bin / best.bin / train_log.csv / status.txt）\n",
           cfg.out_dir);
    if (cfg.lr_decay_every > 0)
        printf("  学习率计划：%.5f 起，每 %d 轮 ×%.2f，下限 %.5f\n",
               cfg.lr, cfg.lr_decay_every, cfg.lr_decay_factor, cfg.lr_min);
    if (cfg.forever) printf("  模式：持续训练（随时可停止，停止时会保存权重）\n");
    else printf("  模式：训练 %d 轮后结束\n", cfg.iterations);
    fflush(stdout);

    /* ---- 固定锚点：拿一个不再变化的网络当尺子，进步曲线不受对手漂移影响 ---- */
    Net   anchor;
    int   have_anchor = 0;
    if (cfg.anchor_path && cfg.anchor_path[0]) {
        net_init_ex(&anchor, cfg.size, cfg.planes, cfg.channels, cfg.vhidden, cfg.blocks, 1);
        if (net_load(&anchor, cfg.anchor_path)) {
            if (anchor.size == cfg.size && anchor.planes == cfg.planes &&
                anchor.channels == cfg.channels) {
                have_anchor = 1;
                printf("  锚点网络：%s（每 %d 轮评估 %d 局，用于低噪声进步曲线）\n",
                       cfg.anchor_path, cfg.anchor_every, cfg.anchor_games);
            } else {
                printf("  锚点网络规格(%d路/%d通道)与当前训练不一致，已忽略\n",
                       anchor.size, anchor.channels);
                net_free(&anchor);
            }
        } else {
            printf("  锚点网络 %s 打不开，已忽略\n", cfg.anchor_path);
            net_free(&anchor);
        }
    }
    int bad_gates = 0;                  /* 连续几次晋级赛表现差 */
    const double t_run0 = goai_now();
    int iter = start_iter;
    int done_iters = 0;

    while (!stop_requested(&cfg) && (cfg.forever || done_iters < cfg.iterations)) {
        const double t_iter0 = goai_now();
        __atomic_store_n(&t.games_done, 0, __ATOMIC_RELAXED);

        ProgressCtx pc;
        pc.t = &t; pc.cfg = &cfg; pc.iter = iter;
        pc.games_target = cfg.games_per_iter;
        pc.t_iter0 = t_iter0; pc.t_run0 = t_run0;
        pc.games_before = total_games; pc.stop = 0;
        pthread_t pth;
        pthread_create(&pth, NULL, progress_thread, &pc);

        int positions = 0;
        const int nthreads = (cfg.threads > 1) ? cfg.threads : 1;
        if (nthreads == 1) {
            ExampleBatch batch;
            if (batch_init(&batch, cfg.max_moves + 2, t.xlen, t.pilen) != 0) return 1;
            for (int g = 0; g < cfg.games_per_iter && !stop_requested(&cfg); g++) {
                Sgf *sgf = NULL;
                if (g < cfg.save_sgf) {
                    snprintf(path, sizeof(path), "%s/games/iter%03d_game%02d.sgf", cfg.out_dir, iter, g + 1);
                    char cmt[160];
                    snprintf(cmt, sizeof(cmt), "GoAI self-play iteration %d game %d", iter, g + 1);
                    sgf = sgf_open(path, cfg.size, cfg.komi, cmt);
                }
                const int n = selfplay_game(&cfg, &s, &t.rng, sgf, &batch);
                if (sgf) sgf_close(sgf);
                if (n < 0) { fprintf(stderr, "self-play failed\n"); return 1; }
                if (n > 0) {
                    pthread_mutex_lock(&t.buf_lock);
                    for (int i = 0; i < batch.n; i++)
                        buffer_push(&t, batch.X + (size_t)i * t.xlen,
                                       batch.PI + (size_t)i * t.pilen, batch.Z[i]);
                    pthread_mutex_unlock(&t.buf_lock);
                }
                positions += n;
                __atomic_add_fetch(&t.games_done, 1, __ATOMIC_RELAXED);
            }
            batch_free(&batch);
        } else {
            SelfPlayJob jobs[64];
            pthread_t   th[64];
            const int   njobs = nthreads < 64 ? nthreads : 64;
            for (int i = 0; i < njobs; i++) {
                memset(&jobs[i], 0, sizeof(jobs[i]));
                jobs[i].cfg = &cfg;
                jobs[i].t = &t;
                jobs[i].iter = iter;
                jobs[i].game_ids = (int *)malloc((size_t)cfg.games_per_iter * sizeof(int));
                search_init(&jobs[i].search, cfg.size, &t.net,
                            cfg.seed + 1000003ULL * (uint64_t)(i + 1) + 7919ULL * (uint64_t)iter,
                            cfg.komi);
                jobs[i].search.pass_min_move = cfg.pass_min_move;
                jobs[i].search.max_moves = cfg.max_moves;
                rng_seed(&jobs[i].rng, cfg.seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)) ^ (uint64_t)iter);
            }
            for (int g = 0; g < cfg.games_per_iter; g++) {
                SelfPlayJob *j = &jobs[g % njobs];
                j->game_ids[j->ngames++] = g;
            }
            for (int i = 0; i < njobs; i++) pthread_create(&th[i], NULL, selfplay_thread, &jobs[i]);
            for (int i = 0; i < njobs; i++) {
                pthread_join(th[i], NULL);
                positions += jobs[i].positions;
                free(jobs[i].game_ids);
                search_free(&jobs[i].search);
            }
        }
        pc.stop = 1;
        pthread_join(pth, NULL);

        if (stop_requested(&cfg)) {
            printf("\n收到停止请求：本轮只跑了部分对局，直接保存当前权重后退出。\n");
            break;
        }

        float pl = 0, vl = 0;
        const double t_sp = goai_now() - t_iter0;   /* 阶段 1：自对弈 */
        const double t_tr0 = goai_now();
        t.lr = train_lr_at_iter(&cfg, iter);      /* 学习率衰减计划 */
        train_gradient_steps(&t, &cfg, cfg.train_steps, &pl, &vl);

        const double t_tr = goai_now() - t_tr0;     /* 阶段 2：梯度训练 */
        const double t_ev0 = goai_now();
        EngineSpec me  = { &t.net, cfg.eval_sims, 0 };
        EngineSpec rnd = { NULL, 0, 1 };
        int wa = 0, wb = 0, dr = 0;
        eval_match(&s, &cfg, me, rnd, cfg.eval_games, &wa, &wb, &dr);
        const double wr = (wa + wb + dr) > 0 ? (double)wa / (double)(wa + wb + dr) : 0.0;

        /* 对固定锚点的胜率：噪声低、随训练单调上升，是判断"到底有没有变强"的主指标 */
        double aw = -1.0;
        if (have_anchor && cfg.anchor_every > 0 && cfg.anchor_games > 0 &&
            (iter % cfg.anchor_every == 0)) {
            EngineSpec me2 = { &t.net, cfg.eval_sims, 0 };
            EngineSpec an2 = { &anchor, cfg.eval_sims, 0 };
            int aa = 0, ab = 0, ad = 0;
            eval_match(&s, &cfg, me2, an2, cfg.anchor_games, &aa, &ab, &ad);
            aw = (aa + ab + ad) > 0 ? (double)aa / (double)(aa + ab + ad) : 0.0;
            printf("  [锚点] 对固定基准胜率 %.1f%% (±%.1f%%，%d 局)\n",
                   aw * 100.0, train_winrate_ci(aw, cfg.anchor_games) * 100.0, cfg.anchor_games);
        }

        snprintf(path, sizeof(path), "%s/latest.bin", cfg.out_dir);
        net_save(&t.net, path);
        snprintf(path, sizeof(path), "%s/iter_%03d.bin", cfg.out_dir, iter);
        net_save(&t.net, path);

        double gw = -1.0;
        if (cfg.gate && (iter % cfg.eval_every == 0)) {
            Net best;
            net_init_ex(&best, cfg.size, cfg.planes, cfg.channels, cfg.vhidden, cfg.blocks, 1);
            if (net_load(&best, best_path)) {
                EngineSpec cur = { &t.net, cfg.eval_sims, 0 };
                EngineSpec bst = { &best,  cfg.eval_sims, 0 };
                int ba = 0, bb = 0, bd = 0;
                eval_match(&s, &cfg, cur, bst, cfg.gate_games, &ba, &bb, &bd);
                gw = (ba + bb + bd) > 0 ? (double)ba / (double)(ba + bb + bd) : 0.0;
                {
                    const double ci = train_winrate_ci(gw, cfg.gate_games) * 100.0;
                    if (gw >= 0.55) {
                        net_save(&t.net, best_path);
                        printf("  [晋级] 对当前最佳胜率 %.1f%% (±%.1f%%) -> 更新 best.bin\n",
                               gw * 100.0, ci);
                    } else {
                        printf("  [保留] 对当前最佳胜率 %.1f%% (±%.1f%%) -> 保留旧 best.bin\n",
                               gw * 100.0, ci);
                    }
                    /* 退化保护：连着几次明显输给旧版 = 练歪了，退回 best 再继续 */
                    if (gw < 0.45) bad_gates++; else bad_gates = 0;
                    if (cfg.rollback && cfg.rollback_patience > 0 &&
                        bad_gates >= cfg.rollback_patience) {
                        printf("  [回滚] 连续 %d 次对最佳胜率 < 45%%，判定退化 -> 载入 best.bin\n",
                               bad_gates);
                        if (net_load(&t.net, best_path)) {
                            memset(t.adam_m, 0, (size_t)t.net.n_params * sizeof(float));
                            memset(t.adam_v, 0, (size_t)t.net.n_params * sizeof(float));
                            t.step = 0;
                            t.lr = train_lr_at_iter(&cfg, iter);
                            printf("  [回滚] 已回到最佳权重，Adam 状态清零，学习率 %.5f\n", t.lr);
                        } else {
                            printf("  [回滚] best.bin 读取失败，保持当前权重\n");
                        }
                        bad_gates = 0;
                    }
                }
            } else {
                net_save(&t.net, best_path);
            }
            net_free(&best);
        }

        const double t_ev = goai_now() - t_ev0;     /* 阶段 3：评估/晋级/锚点 */
        total_games += cfg.games_per_iter;
        const double el = goai_now() - t_iter0;
        const double total_el = goai_now() - t_run0;
        printf("──────────────────────────────────────────────────────────────\n");
        printf("  第 %d 轮完成 · 自对弈 %d 局 / %d 局面 · 用时 %.1fs\n",
               iter, cfg.games_per_iter, positions, el);
        if (el > 0.0)
            printf("  耗时分布：自对弈 %.1fs (%.0f%%) · 训练 %.1fs (%.0f%%) · 评估 %.1fs (%.0f%%)\n",
                   t_sp, 100.0 * t_sp / el, t_tr, 100.0 * t_tr / el, t_ev, 100.0 * t_ev / el);
        printf("  策略损失 %.4f · 价值损失 %.4f · 对随机胜率 %.1f%%",
               pl, vl, wr * 100.0);
        printf(" · 学习率 %.5f", t.lr);
        if (gw >= 0.0) printf(" · 对最佳 %.1f%%", gw * 100.0);
        printf("\n");
        printf("  累计 %ld 局 · 已训练 %.1f 分钟 · 权重 %s/latest.bin\n",
               total_games, total_el / 60.0, cfg.out_dir);
        printf("──────────────────────────────────────────────────────────────\n");
        fflush(stdout);

        if (csv) {
            fprintf(csv, "%d,%d,%d,%.5f,%.5f,%.4f,%.4f,%.4f,%.2f,%.6f,%.4f,%.2f,%.2f,%.2f\n",
                    iter, cfg.games_per_iter, positions, pl, vl, net_param_norm(&t.net), wr, gw, el,
                    t.lr, aw, t_sp, t_tr, t_ev);
            fflush(csv);
        }
        {
            snprintf(path, sizeof(path), "%s/state.txt", cfg.out_dir);
            FILE *sf = fopen(path, "w");
            if (sf) { fprintf(sf, "%d %ld\n", iter + 1, total_games); fclose(sf); }
        }
        write_status(&cfg, (cfg.forever || done_iters + 1 < cfg.iterations) ? "running" : "finished",
                     iter + 1, 0, el, total_el, total_games, "winrate_vs_random", wr);

        if (cfg.plot) {
            char cmd[1400];
            snprintf(cmd, sizeof(cmd),
                     "python3 tools/plot.py '%s/train_log.csv' '%s/training_curve.png' >/dev/null 2>&1",
                     cfg.out_dir, cfg.out_dir);
            if (system(cmd) != 0) { /* plotting is best effort */ }
        }
        iter++;
        done_iters++;
    }

    snprintf(path, sizeof(path), "%s/latest.bin", cfg.out_dir);
    net_save(&t.net, path);
    const double total_el = goai_now() - t_run0;
    write_status(&cfg, g_stop ? "stopped" : "finished", iter, 0, 0.0, total_el, total_games,
                 NULL, 0.0);
    if (g_stop) printf("\n训练已停止（共 %d 轮 / %ld 局 / %.1f 分钟），权重保存在 %s/latest.bin\n",
                       done_iters, total_games, total_el / 60.0, cfg.out_dir);
    else printf("\n训练结束（共 %d 轮 / %ld 局 / %.1f 分钟）。\n", done_iters, total_games, total_el / 60.0);
    fflush(stdout);

    if (csv) fclose(csv);
    if (have_anchor) net_free(&anchor);
    search_free(&s);
    trainer_free(&t);
    if (cfg.plot) {
        char cmd[1400];
        snprintf(cmd, sizeof(cmd),
                 "python3 tools/plot.py '%s/train_log.csv' '%s/training_curve.png' >/dev/null 2>&1",
                 cfg.out_dir, cfg.out_dir);
        if (system(cmd) != 0) { }
    }
    return 0;
}

int train_bench(const TrainConfig *cfg) {
    TrainConfig cfgl = *cfg;
    if (cfgl.threads <= 0) {
        long ncpu = goai_cpu_count();
        cfgl.threads = (ncpu > 0) ? (int)ncpu : 1;
        if (cfgl.threads > 8) cfgl.threads = 8;
    }
    cfg = &cfgl;

    Trainer t;
    if (trainer_init(&t, cfg, cfg->seed) != 0) return 1;
    Search s;
    search_init(&s, cfg->size, &t.net, 1, cfg->komi);
    s.pass_min_move = cfg->pass_min_move;
    s.max_moves = cfg->max_moves;

    float x[BOARD_MAX_POINTS * 8];
    Board b;
    board_init(&b, cfg->size);
    net_features(&b, x);
    float pol[BOARD_MAX_POINTS + 1], v;
    const int N = 2000;
    double t0 = goai_now();
    for (int i = 0; i < N; i++) net_forward(&t.net, &t.cache, x, pol, &v);
    double dtn = goai_now() - t0;
    printf("network : %d params, %.3f ms/eval (%.1f evals/s)\n",
           t.net.n_params, dtn / N * 1e3, N / dtn);

    t0 = goai_now();
    const int sims = 200;
    for (int i = 0; i < 20; i++) {
        board_init(&b, cfg->size);
        search_run(&s, &b, sims, 0.0f, 0.0f, 0.0f, pol, &v);
    }
    double dts = goai_now() - t0;
    printf("mcts    : %.3f ms per %d sims => %.0f sims/s\n", dts / 20 * 1e3, sims, sims * 20 / dts);

    t0 = goai_now();
    const int games = 3;
    int total_moves = 0;
    for (int g = 0; g < games; g++) {
        Board bd;
        board_init(&bd, cfg->size);
        while (bd.passes < 2 && bd.nmoves < cfg->max_moves) {
            const int mv = search_run(&s, &bd, cfg->sims, 0.0f, 0.0f, 0.0f, pol, &v);
            board_play(&bd, mv);
            total_moves++;
        }
    }
    double dtg = goai_now() - t0;
    printf("selfplay: %.2f s/game (%d sims/move, %.1f moves/game) => %.1f games/min\n",
           dtg / games, cfg->sims, (double)total_moves / games, 60.0 * games / dtg);

    search_free(&s);
    trainer_free(&t);
    return 0;
}
