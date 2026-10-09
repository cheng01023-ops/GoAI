/* gplay.c - C 引擎批量自对弈 + GPU 服务端批量推理
 *
 * 每线程并行 slots 盘棋：每轮每盘各做一次模拟，把这一批叶子（threads*slots 个）
 * 拼成请求发往 GPU 服务，拿回策略/价值后展开回传。这样显卡每次能拿到几百个局面。
 * 样本通过 upload 通道回传，由服务端在 GPU 上训练。
 */
#include "gplay.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "mcts.h"
#include "net.h"

typedef struct {
    const void *cfg;
    RemoteNet  *rc;
    int         nslots;
    Search     *searches;
    Board      *boards;
    Board      *backups;
    int        *active;
    int        *nmoves;
    int        *winner;
    float      *xbuf;      /* [nslots][planes*nn] */
    float      *pbuf;      /* [nslots][nn+1] */
    float      *vbuf;      /* [nslots] */
    int        *leaf;
    int        *slot_of;
    int        *ex_game;   /* 样本属于哪盘棋 */
    int        *ex_ply;    /* 样本是第几手 */
    ExampleBatch batch;
    Rng         rng;
    int         planes, nn, pnn;
    int         sims, max_moves, pass_min_move, temp_moves, size;
    double      komi;
    int         save_sgf;
    const char *out_dir;
    int         tid;
    int         iter;
    long        positions;
    int         games_done;
} GWorker;

static volatile int g_quit = 0;

/* 停止条件：出现 <out>/STOP 文件 */
static int stop_now(const char *out_dir) {
    char p[1200];
    snprintf(p, sizeof(p), "%s/STOP", out_dir);
    return goai_file_exists(p);
}

typedef struct {
    GWorker   *workers;
    int        nthreads;
    pthread_t *th;
    RemoteNet *rcs;         /* 每个线程一条连接 */
} GPool;

/* 写 SGF */
typedef struct { FILE *f; int size; int n; } GSgf;

static GSgf *gsgf_open(const char *path, int size, double komi, const char *cmt) {
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    GSgf *g = (GSgf *)calloc(1, sizeof(GSgf));
    g->f = f; g->size = size;
    fprintf(f, "(;GM[1]FF[4]CA[UTF-8]AP[GoAI-GPU:1.0]SZ[%d]KM[%.1f]PB[Black]PW[White]RU[Chinese]C[%s]\n",
            size, komi, cmt);
    return g;
}
static void gsgf_move(GSgf *g, int move, int color) {
    if (!g) return;
    if (move == M_PASS) fprintf(g->f, ";%c[]", color == 1 ? 'B' : 'W');
    else fprintf(g->f, ";%c[%c%c]", color == 1 ? 'B' : 'W',
                 'a' + (move % g->size), 'a' + (move / g->size));
    if (++g->n % 10 == 0) fprintf(g->f, "\n");
}
static void gsgf_close(GSgf *g, int winner) {
    if (!g) return;
    fprintf(g->f, "RE[%s])\n", winner == 1 ? "B+R" : (winner == 2 ? "W+R" : "0"));
    fclose(g->f);
    free(g);
}

static void *gplay_thread(void *arg) {
    GWorker *w = (GWorker *)arg;
    GSgf *sgf = NULL;
    if (w->save_sgf && w->nslots > 0) {
        char path[1024], cmt[160];
        snprintf(path, sizeof(path), "%s/games/gpu_t%02d.sgf", w->out_dir, w->tid);
        snprintf(cmt, sizeof(cmt), "GoAI GPU self-play thread %d", w->tid);
        sgf = gsgf_open(path, w->size, w->komi, cmt);
    }
    for (int g = 0; g < w->nslots; g++) {
        w->active[g] = g;
        w->nmoves[g] = 0;
        w->winner[g] = 0;
    }
    int nactive = w->nslots;
    while (nactive > 0 && !g_quit) {
        if (stop_now(w->out_dir)) break;
        /* 每一手都从当前局面重建搜索树（C 侧没有树复用，保证根与棋盘一致） */
        for (int i = 0; i < nactive; i++) {
            const int g = w->active[i];
            search_begin_move(&w->searches[g]);
        }
        for (int sim = 0; sim < w->sims && !g_quit; sim++) {
            int cnt = 0;
            for (int i = 0; i < nactive; i++) {
                const int g = w->active[i];
                w->backups[g] = w->boards[g];
                int is_term = 0;
                int lf = search_select_leaf(&w->searches[g], &w->boards[g], &is_term);
                if (is_term) {
                    search_apply_terminal(&w->searches[g], lf, w->searches[g].nodes[lf].terminal_value);
                    w->boards[g] = w->backups[g];
                    continue;
                }
                net_features(&w->boards[g], w->xbuf + (size_t)cnt * w->pnn);
                w->leaf[cnt] = lf;
                w->slot_of[cnt] = g;
                cnt++;
            }
            if (cnt > 0 && w->rc) {
                if (remote_eval(w->rc, cnt, w->xbuf, w->pbuf, w->vbuf) != 0) { g_quit = 1; break; }
                for (int i = 0; i < cnt; i++) {
                    const int g = w->slot_of[i];
                    search_apply_leaf(&w->searches[g], w->leaf[i], &w->boards[g],
                                      w->pbuf + (size_t)i * (w->nn + 1), w->vbuf[i]);
                    w->boards[g] = w->backups[g];
                }
            }
        }
        if (g_quit) break;
        int still = 0;
        for (int i = 0; i < nactive; i++) {
            const int g = w->active[i];
            const float temp = (w->nmoves[g] < w->temp_moves) ? 1.0f : 0.0f;
            const int mv = search_choose_move(&w->searches[g], temp, &w->rng);
            float pol[BOARD_MAX_POINTS + 1];
            search_root_policy(&w->searches[g], pol);
            float xbuf[BOARD_MAX_POINTS * 8];
            net_features(&w->boards[g], xbuf);
            if (w->batch.n < w->batch.cap) {
                uint8_t *dst = w->batch.X + (size_t)w->batch.n * w->batch.xlen;
                for (int k = 0; k < w->batch.xlen; k++) dst[k] = xbuf[k] > 0.5f ? 1 : 0;
                memcpy(w->batch.PI + (size_t)w->batch.n * w->batch.pilen, pol,
                       (size_t)w->batch.pilen * sizeof(float));
                w->ex_game[w->batch.n] = g;
                w->ex_ply[w->batch.n] = w->nmoves[g];
                w->batch.n++;
            }
            if (sgf && g == w->active[0]) gsgf_move(sgf, mv, w->boards[g].to_move);
            if (!board_play(&w->boards[g], mv)) board_play(&w->boards[g], M_PASS);   /* 兜底 */
            w->searches[g].pending_advance = mv;   /* 树复用 */
            w->nmoves[g]++;
            w->positions++;
            if (w->boards[g].passes >= 2 || w->boards[g].nmoves >= w->max_moves) {
                w->winner[g] = board_winner(&w->boards[g], w->komi);
                w->games_done++;
                continue;
            }
            w->active[still++] = g;
        }
        nactive = still;
    }
    if (sgf) gsgf_close(sgf, w->winner[0]);
    return NULL;
}

int gplay_run(const char *addr, int threads, int slots_per_thread, int sims,
              int size, double komi, int max_moves, int pass_min_move,
              int temp_moves, uint64_t seed, const char *out_dir, int save_sgf) {
    char host[128];
    int port = 8899;
    if (remote_parse_addr(addr, host, sizeof(host), &port) != 0) return 1;

    RemoteNet probe;
    if (remote_connect(&probe, host, port) != 0) return 1;
    const int planes = probe.planes, nn = probe.nn;
    printf("=== GoAI GPU 自对弈（C 搜索 + GPU 批量推理）===\n");
    printf("  服务端：%s:%d | 棋盘 %dx%d | %d 通道 | 单次最多合并 %d 个局面\n",
           host, port, probe.size, probe.size, probe.channels, probe.max_batch);
    printf("  线程 %d × 每线程 %d 盘 = %d 盘并行 | 每步 %d 次模拟 | 训练在服务端 GPU 上进行\n",
           threads, slots_per_thread, threads * slots_per_thread, sims);

    GPool pool;
    pool.nthreads = threads;
    pool.workers = (GWorker *)calloc(threads, sizeof(GWorker));
    pool.th = (pthread_t *)calloc(threads, sizeof(pthread_t));
    pool.rcs = (RemoteNet *)calloc(threads, sizeof(RemoteNet));
    pool.rcs[0] = probe;
    for (int i = 1; i < threads; i++) {
        if (remote_connect(&pool.rcs[i], host, port) != 0) return 1;
    }

    char gdir[1200];
    snprintf(gdir, sizeof(gdir), "%s/games", out_dir);
    goai_mkdir_p(gdir);

    for (int i = 0; i < threads; i++) {
        GWorker *w = &pool.workers[i];
        memset(w, 0, sizeof(*w));
        w->rc = &pool.rcs[i];
        w->nslots = slots_per_thread;
        w->planes = planes; w->nn = nn; w->pnn = planes * nn;
        w->sims = sims; w->size = size; w->komi = komi;
        w->max_moves = max_moves; w->pass_min_move = pass_min_move; w->temp_moves = temp_moves;
        w->out_dir = out_dir; w->save_sgf = save_sgf; w->tid = i;
        w->searches = (Search *)calloc(slots_per_thread, sizeof(Search));
        w->boards = (Board *)calloc(slots_per_thread, sizeof(Board));
        w->backups = (Board *)calloc(slots_per_thread, sizeof(Board));
        w->active = (int *)calloc(slots_per_thread, sizeof(int));
        w->nmoves = (int *)calloc(slots_per_thread, sizeof(int));
        w->winner = (int *)calloc(slots_per_thread, sizeof(int));
        w->xbuf = (float *)calloc((size_t)slots_per_thread * w->pnn, sizeof(float));
        w->pbuf = (float *)calloc((size_t)slots_per_thread * (nn + 1), sizeof(float));
        w->vbuf = (float *)calloc((size_t)slots_per_thread, sizeof(float));
        w->leaf = (int *)calloc(slots_per_thread, sizeof(int));
        w->slot_of = (int *)calloc(slots_per_thread, sizeof(int));
        if (batch_init(&w->batch, 4096, planes * nn, nn + 1) != 0) return 1;
        w->ex_game = (int *)calloc(4096, sizeof(int));
        w->ex_ply = (int *)calloc(4096, sizeof(int));
        rng_seed(&w->rng, seed + 7919ULL * (uint64_t)(i + 1));
        for (int g = 0; g < slots_per_thread; g++) {
            board_init(&w->boards[g], size);
            search_init(&w->searches[g], size, NULL, seed + 104729ULL * (uint64_t)(i * 100 + g), komi);
            w->searches[g].pass_min_move = pass_min_move;
            w->searches[g].max_moves = max_moves;
        }
    }

    const double t0 = goai_now();
    int iter = 0;
    while (!g_quit) {
        iter++;
        for (int i = 0; i < threads; i++) {
            GWorker *w = &pool.workers[i];
            w->iter = iter;
            w->batch.n = 0;
            w->positions = 0;
            for (int g = 0; g < slots_per_thread; g++) {
                board_init(&w->boards[g], size);
                search_free(&w->searches[g]);
                search_init(&w->searches[g], size, NULL, seed + 104729ULL * (uint64_t)(iter * 1000 + i * 100 + g), komi);
                w->searches[g].pass_min_move = pass_min_move;
                w->searches[g].max_moves = max_moves;
            }
        }
        for (int i = 0; i < threads; i++) pthread_create(&pool.th[i], NULL, gplay_thread, &pool.workers[i]);
        for (int i = 0; i < threads; i++) pthread_join(pool.th[i], NULL);

        long pos = 0, games = 0;
        for (int i = 0; i < threads; i++) {
            /* 标注 z 并上传样本（z 从该局面行棋方视角） */
            GWorker *w = &pool.workers[i];
            for (int k = 0; k < w->batch.n; k++) {
                const int g = w->ex_game[k];
                const int wint = (g >= 0 && g < w->nslots) ? w->winner[g] : 0;
                const int player = (w->ex_ply[k] % 2 == 0) ? 1 : 2;
                w->batch.Z[k] = (wint == 0) ? 0.0f : (wint == player ? 1.0f : -1.0f);
            }
            if (w->batch.n > 0 && w->rc) {
                remote_upload(w->rc, w->batch.n, w->batch.X, w->batch.PI, w->batch.Z);
            }
            pos += w->positions;
            games += w->games_done;
        }
        const double el = goai_now() - t0;
        /* 状态文件（与 C 版训练器同格式，方便同一套监控脚本） */
        char spath[1200];
        snprintf(spath, sizeof(spath), "%s/status.txt", out_dir);
        FILE *sf = fopen(spath, "w");
        if (sf) {
            char ts[64];
            goai_localtime_str(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S");
            fprintf(sf, "state=running\niteration=%d\ngame=0\ngames_this_iter=%ld\n", iter, games);
            fprintf(sf, "total_games=%ld\npositions=%ld\n", games, pos);
            fprintf(sf, "elapsed_iter=%.1f\nelapsed_total=%.1f\n", el, el);
            fprintf(sf, "threads=%d\nslots=%d\nsims=%d\nchannels=%d\n",
                    threads, slots_per_thread, sims, probe.channels);
            fprintf(sf, "pid=%ld\nmode=gpu-server\nupdated=%s\n", goai_pid(), ts);
            fclose(sf);
        }
        printf("第 %d 轮 | %d 盘并行 | 本批 %ld 局面 | 累计 %.1f 分钟 | %.0f 局面/秒\n",
               iter, threads * slots_per_thread, pos, el / 60.0, (double)pos / (el > 0 ? el : 1));
        fflush(stdout);
        if (stop_now(out_dir)) break;
    }
    for (int i = 0; i < threads; i++) {
        batch_free(&pool.workers[i].batch);
        free(pool.workers[i].searches); free(pool.workers[i].boards); free(pool.workers[i].backups);
        free(pool.workers[i].active); free(pool.workers[i].nmoves); free(pool.workers[i].winner);
        free(pool.workers[i].xbuf); free(pool.workers[i].pbuf); free(pool.workers[i].vbuf);
        free(pool.workers[i].leaf); free(pool.workers[i].slot_of);
        free(pool.workers[i].ex_game); free(pool.workers[i].ex_ply);
        remote_close(&pool.rcs[i]);
    }
    free(pool.workers); free(pool.th); free(pool.rcs);
    printf("GPU 自对弈结束（收到停止请求）。\n");
    return 0;
}
