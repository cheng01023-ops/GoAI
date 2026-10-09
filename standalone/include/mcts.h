/* mcts.h - AlphaZero-style PUCT search (and a random-rollout baseline mode) */
#ifndef GOAI_MCTS_H
#define GOAI_MCTS_H

#include "board.h"
#include "net.h"
#include "rand.h"

typedef struct {
    int   move;
    float prior;
    int   child;      /* index of the child node, -1 if not created yet */
    int   visits;
    float value_sum;  /* values from the perspective of the player to move at the CHILD */
} Edge;

typedef struct {
    int      first_edge;
    int      n_edges;
    int      visits;
    int      move;          /* move leading here (M_NONE for the root) */
    float    prior;
    uint8_t  expanded;
    uint8_t  terminal;
    float    terminal_value;
} Node;

typedef struct {
    int      size, nn;
    Node    *nodes;
    Edge    *edges;
    int      n_nodes, cap_nodes, n_edges, cap_edges;
    const Net *net;         /* NULL => random-rollout mode */
    NetCache cache;
    int      cache_ready;   /* cache is initialised for s->net */
    int      x_buf_len;
    float   *x_buf;
    float   *policy_buf;
    float    c_puct;
    int      rollout_moves;
    double   komi;
    int      max_moves;     /* game length cap for terminal detection */
    int      pass_min_move; /* passes are not offered before this move number */
    Rng      rng;
    double   root_value;    /* mean value at the root, root player's view */
    /* 两阶段接口用的路径记录 */
    struct { int node; int edge; } path[BOARD_MAX_POINTS * 2 + 8];
    int      path_len;
    /* 可选的进度回调（界面显示“AI 思考中”进度条用） */
    volatile int *progress_done;
    int           progress_total;
} Search;

void search_init(Search *s, int size, const Net *net, uint64_t seed, double komi);
void search_free(Search *s);

/* switch networks; cache and buffers are re-sized when the shape changes */
void search_set_net(Search *s, const Net *net);

/* Run `sims` playouts from b0. Returns the chosen move (temperature-sampled),
   fills policy_out (nn+1 visit distribution, temperature applied) and the root value. */
int  search_run(Search *s, const Board *b0, int sims, float dirichlet_alpha, float noise_eps,
                float temperature, float *policy_out, float *root_value_out);

/* ---- 两阶段接口：先选出叶子，批量评估后再展开回传（GPU 批量推理用） ----
   返回叶子节点索引；沿途着法会落在 b 上；
   *is_terminal=1 时表示该局面已终局（此时 *terminal_value 给出从行棋方视角的价值） */
/* 每手开始前重置搜索树（根节点对应当前局面） */
void search_begin_move(Search *s);
int  search_select_leaf(Search *s, Board *b, int *is_terminal);
/* 用外部算好的 policy/value 展开叶子并回传 */
void search_apply_leaf(Search *s, int leaf, const Board *b, const float *policy, float value);
/* 终局叶子的回传 */
void search_apply_terminal(Search *s, int leaf, float value);
/* 选点（温度<=0 取访问最多） */
int  search_choose_move(Search *s, float temperature, Rng *rng);
/* 根节点的访问次数分布（训练用的策略目标），长度为 nn+1 */
void search_root_policy(const Search *s, float *policy_out);

/* visit counts of the last search, without temperature */
void search_root_visits(const Search *s, int *out_counts);

#endif /* GOAI_MCTS_H */
