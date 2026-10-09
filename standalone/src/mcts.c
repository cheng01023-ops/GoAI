/* mcts.c - PUCT Monte-Carlo tree search.
 *
 * Two evaluation modes share the same tree:
 *   - net mode     : priors and values come from the policy/value network (AlphaZero)
 *   - rollout mode : uniform priors, values from a random playout (classic MCTS baseline)
 */
#include "mcts.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void search_init(Search *s, int size, const Net *net, uint64_t seed, double komi) {
    memset(s, 0, sizeof(*s));
    s->size = size;
    s->nn = size * size;
    s->net = net;
    s->c_puct = net ? 1.5f : 2.0f;
    s->rollout_moves = 3 * size * size;
    s->komi = komi;
    s->max_moves = 3 * size * size;
    s->pass_min_move = 2 * size;
    s->pending_advance = -1;
    rng_seed(&s->rng, seed ? seed : 0xBADC0FFEEULL);
    s->cap_nodes = 4096;
    s->cap_edges = 4096;
    s->nodes = (Node *)malloc((size_t)s->cap_nodes * sizeof(Node));
    s->edges = (Edge *)malloc((size_t)s->cap_edges * sizeof(Edge));
    s->x_buf_len = (net ? net->planes : 1) * s->nn;
    s->x_buf = (float *)calloc((size_t)s->x_buf_len, sizeof(float));
    s->policy_buf = (float *)calloc((size_t)s->nn + 1, sizeof(float));
    if (!s->nodes || !s->edges || !s->x_buf || !s->policy_buf) {
        fprintf(stderr, "search_init: out of memory\n");
        exit(1);
    }
    if (net) { net_cache_init(net, &s->cache); s->cache_ready = 1; }
}

void search_free(Search *s) {
    free(s->nodes);
    free(s->edges);
    free(s->x_buf);
    free(s->policy_buf);
    if (s->cache_ready) net_cache_free(&s->cache);
    memset(s, 0, sizeof(*s));
}

void search_set_net(Search *s, const Net *net) {
    if (s->net == net) return;
    if (s->cache_ready) { net_cache_free(&s->cache); s->cache_ready = 0; }
    s->net = net;
    if (!net) { s->c_puct = 2.0f; return; }
    const int need = net->planes * net->nn;
    if (s->x_buf_len < need) {
        free(s->x_buf);
        s->x_buf = (float *)calloc((size_t)need, sizeof(float));
        if (!s->x_buf) { fprintf(stderr, "search_set_net: out of memory\n"); exit(1); }
        s->x_buf_len = need;
    }
    net_cache_init(net, &s->cache);
    s->cache_ready = 1;
    s->c_puct = 1.5f;
}

static int new_node(Search *s, int move, float prior) {
    if (s->n_nodes == s->cap_nodes) {
        s->cap_nodes *= 2;
        s->nodes = (Node *)realloc(s->nodes, (size_t)s->cap_nodes * sizeof(Node));
        if (!s->nodes) { fprintf(stderr, "mcts: realloc failed\n"); exit(1); }
    }
    Node *n = &s->nodes[s->n_nodes];
    memset(n, 0, sizeof(*n));
    n->move = move;
    n->prior = prior;
    return s->n_nodes++;
}

static void reserve_edges(Search *s, int extra) {
    if (s->n_edges + extra <= s->cap_edges) return;
    while (s->cap_edges < s->n_edges + extra) s->cap_edges *= 2;
    s->edges = (Edge *)realloc(s->edges, (size_t)s->cap_edges * sizeof(Edge));
    if (!s->edges) { fprintf(stderr, "mcts: realloc failed\n"); exit(1); }
}

static void search_reset(Search *s) {
    s->root = 0;
    s->n_nodes = 0;
    s->n_edges = 0;
    new_node(s, M_NONE, 1.0f);
}

static float rollout_value(Search *s, const Board *b) {
    Board tmp = *b;
    return board_playout_value(&tmp, &s->rng, s->rollout_moves, s->komi);
}

/* expand the node and return the value from the point of view of the player to move */
static float expand_and_evaluate(Search *s, const Board *b, int node_idx) {
    Node *n = &s->nodes[node_idx];
    n->expanded = 1;

    if (b->passes >= 2 || b->nmoves >= s->max_moves) {
        const int w = board_winner(b, s->komi);
        n->terminal = 1;
        n->terminal_value = (w == 0) ? 0.0f : (w == b->to_move ? 1.0f : -1.0f);
        return n->terminal_value;
    }

    int moves[BOARD_MAX_POINTS + 1];
    bool allow_pass = b->nmoves >= s->pass_min_move;
    int nm = board_legal_moves(b, moves, allow_pass);
    if (nm == 0) {
        allow_pass = true;
        nm = board_legal_moves(b, moves, allow_pass);
    }

    float value = 0.0f;
    if (s->net) {
        net_features(b, s->x_buf);
        net_forward(s->net, &s->cache, s->x_buf, s->policy_buf, &value);
    }

    reserve_edges(s, nm);
    n->first_edge = s->n_edges;
    n->n_edges = nm;
    float total = 0.0f;
    for (int i = 0; i < nm; i++) {
        Edge *e = &s->edges[s->n_edges + i];
        e->move = moves[i];
        e->child = -1;
        e->visits = 0;
        e->value_sum = 0.0f;
        if (s->net) {
            float p = s->policy_buf[board_policy_index(b, moves[i])];
            if (!(p > 1e-8f)) p = 1e-8f;
            e->prior = p;
            total += p;
        } else {
            e->prior = 1.0f;
            total += 1.0f;
        }
    }
    for (int i = 0; i < nm; i++) s->edges[n->first_edge + i].prior /= total;
    s->n_edges += nm;

    if (!s->net) value = rollout_value(s, b);
    return value;
}

static float simulate(Search *s, Board *b, int node_idx) {
    Node *n = &s->nodes[node_idx];
    if (n->terminal) return n->terminal_value;
    if (!n->expanded) return expand_and_evaluate(s, b, node_idx);

    const float sqrt_n = sqrtf((float)(n->visits > 0 ? n->visits : 1));
    float best = -1e30f;
    int   best_i = 0;
    for (int i = 0; i < n->n_edges; i++) {
        const Edge *e = &s->edges[n->first_edge + i];
        const float q = e->visits > 0 ? -(e->value_sum / (float)e->visits) : 0.0f;
        const float u = s->c_puct * e->prior * sqrt_n / (1.0f + (float)e->visits);
        const float sc = q + u;
        if (sc > best) { best = sc; best_i = i; }
    }
    Edge *e = &s->edges[n->first_edge + best_i];
    if (e->child < 0) e->child = new_node(s, e->move, e->prior);

    Board save = *b;
    if (!board_play(b, e->move)) { *b = save; }   /* cannot happen: only legal moves are stored */
    const float v = simulate(s, b, e->child);
    *b = save;

    e->visits++;
    e->value_sum += v;
    n->visits++;
    return -v;
}

/* ---------------- 两阶段接口（供 GPU 批量推理的驱动使用） ---------------- */

void search_begin_move(Search *s) { search_reset(s); }

static float leaf_terminal_value(Search *s, const Board *b) {
    const int w = board_winner(b, s->komi);
    return (w == 0) ? 0.0f : (w == b->to_move ? 1.0f : -1.0f);
}

int search_select_leaf(Search *s, Board *b, int *is_terminal) {
    int node = 0;
    s->path_len = 0;
    for (;;) {
        Node *n = &s->nodes[node];
        if (n->terminal) {
            *is_terminal = 1;
            return node;
        }
        if (!n->expanded) {
            /* 未展开的叶子也可能是终局局面 */
            if (b->passes >= 2 || b->nmoves >= s->max_moves) {
                n->terminal = 1;
                n->terminal_value = leaf_terminal_value(s, b);
                *is_terminal = 1;
                return node;
            }
            *is_terminal = 0;
            return node;
        }
        const float sqrt_n = sqrtf((float)(n->visits > 0 ? n->visits : 1));
        float best = -1e30f;
        int   best_i = 0;
        for (int i = 0; i < n->n_edges; i++) {
            const Edge *e = &s->edges[n->first_edge + i];
            const float q = e->visits > 0 ? -(e->value_sum / (float)e->visits) : 0.0f;
            const float u = s->c_puct * e->prior * sqrt_n / (1.0f + (float)e->visits);
            const float sc = q + u;
            if (sc > best) { best = sc; best_i = i; }
        }
        Edge *e = &s->edges[n->first_edge + best_i];
        if (e->child < 0) e->child = new_node(s, e->move, e->prior);
        board_play(b, e->move);
        if (s->path_len < (int)(sizeof(s->path) / sizeof(s->path[0]))) {
            s->path[s->path_len].node = node;
            s->path[s->path_len].edge = best_i;
            s->path_len++;
        }
        node = e->child;
    }
}

void search_apply_terminal(Search *s, int leaf, float value) {
    Node *n = &s->nodes[leaf];
    n->expanded = 1;
    n->terminal = 1;
    n->terminal_value = value;
    n->visits++;
    float v = value;
    for (int i = s->path_len - 1; i >= 0; i--) {
        Node *pn = &s->nodes[s->path[i].node];
        Edge *e = &s->edges[pn->first_edge + s->path[i].edge];
        e->visits++;
        e->value_sum += v;
        pn->visits++;
        v = -v;
    }
}

void search_apply_leaf(Search *s, int leaf, const Board *b, const float *policy, float value) {
    Node *n = &s->nodes[leaf];
    n->expanded = 1;
    n->visits++;
    int moves[BOARD_MAX_POINTS + 1];
    bool allow_pass = b->nmoves >= s->pass_min_move;
    int nm = board_legal_moves(b, moves, allow_pass);
    if (nm == 0) {
        allow_pass = true;
        nm = board_legal_moves(b, moves, allow_pass);
    }
    reserve_edges(s, nm);
    n->first_edge = s->n_edges;
    n->n_edges = nm;
    float total = 0.0f;
    for (int i = 0; i < nm; i++) {
        Edge *e = &s->edges[s->n_edges + i];
        e->move = moves[i];
        e->child = -1;
        e->visits = 0;
        e->value_sum = 0.0f;
        float p = policy[board_policy_index(b, moves[i])];
        if (!(p > 1e-8f)) p = 1e-8f;
        e->prior = p;
        total += p;
    }
    for (int i = 0; i < nm; i++) s->edges[n->first_edge + i].prior /= total;
    s->n_edges += nm;
    /* 回传 */
    float v = value;
    for (int i = s->path_len - 1; i >= 0; i--) {
        Node *pn = &s->nodes[s->path[i].node];
        Edge *e = &s->edges[pn->first_edge + s->path[i].edge];
        e->visits++;
        e->value_sum += v;
        pn->visits++;
        v = -v;
    }
}

/* 树复用：把根前进到着法 move 对应的子节点。
   返回 1 成功（该子节点已展开，可以接着搜），0 表示无法复用（调用方应重建）。 */
int search_advance_root(Search *s, int move) {
    if (move == M_PASS) return 0;
    const Node *r = &s->nodes[s->root];
    if (!r->expanded) return 0;
    for (int i = 0; i < r->n_edges; i++) {
        const Edge *e = &s->edges[r->first_edge + i];
        if (e->move != move) continue;
        if (e->child < 0 || !s->nodes[e->child].expanded) return 0;
        if (s->nodes[e->child].terminal) return 0;
        s->root = e->child;
        return 1;
    }
    return 0;
}
int search_choose_move(Search *s, float temperature, Rng *rng) {
    Node *root = &s->nodes[s->root];
    if (root->n_edges <= 0) return M_PASS;
    if (temperature <= 1e-3f) {
        int best_visits = -1, best_move = s->edges[root->first_edge].move;
        for (int i = 0; i < root->n_edges; i++) {
            const Edge *e = &s->edges[root->first_edge + i];
            if (e->visits > best_visits) { best_visits = e->visits; best_move = e->move; }
        }
        return best_move;
    }
    double total = 0.0;
    static _Thread_local double w[BOARD_MAX_POINTS + 1];
    for (int i = 0; i < root->n_edges; i++) {
        const Edge *e = &s->edges[root->first_edge + i];
        w[i] = pow((double)e->visits, 1.0 / (double)temperature);
        total += w[i];
    }
    if (total <= 0) return s->edges[root->first_edge].move;
    double r = rng_double(rng) * total, acc = 0.0;
    for (int i = 0; i < root->n_edges; i++) {
        acc += w[i];
        if (r < acc) return s->edges[root->first_edge + i].move;
    }
    return s->edges[root->first_edge + root->n_edges - 1].move;
}

void search_root_policy(const Search *s, float *policy_out) {
    const Node *root = &s->nodes[s->root];
    const int nn = s->nn;
    for (int i = 0; i <= nn; i++) policy_out[i] = 0.0f;
    double total = 0.0;
    Board tmp;
    board_init(&tmp, s->size);
    for (int i = 0; i < root->n_edges; i++) {
        const Edge *e = &s->edges[root->first_edge + i];
        policy_out[board_policy_index(&tmp, e->move)] = (float)e->visits;
        total += (double)e->visits;
    }
    if (total > 0) for (int i = 0; i <= nn; i++) policy_out[i] = (float)((double)policy_out[i] / total);
    else policy_out[board_policy_index(&tmp, root->n_edges > 0 ? s->edges[root->first_edge].move : M_PASS)] = 1.0f;
}

static void add_dirichlet_noise(Search *s, double alpha, double eps) {
    Node *root = &s->nodes[s->root];
    if (root->n_edges <= 0) return;
    float *noise = (float *)malloc((size_t)root->n_edges * sizeof(float));
    double sum = 0;
    for (int i = 0; i < root->n_edges; i++) {
        noise[i] = (float)rng_gamma(&s->rng, alpha);
        sum += noise[i];
    }
    if (sum <= 0) sum = 1;
    for (int i = 0; i < root->n_edges; i++) {
        Edge *e = &s->edges[root->first_edge + i];
        e->prior = (float)((1.0 - eps) * e->prior + eps * (noise[i] / sum));
    }
    free(noise);
}

int search_run(Search *s, const Board *b0, int sims, float dirichlet_alpha, float noise_eps,
               float temperature, float *policy_out, float *root_value_out) {
    /* 树复用：调用方走完一手后会设置 pending_advance，
       这里优先前进到那棵子树；否则正常重建。 */
    if (s->pending_advance >= 0) {
        if (!search_advance_root(s, s->pending_advance)) search_reset(s);
        s->pending_advance = -1;
    } else {
        search_reset(s);
    }
    Board b = *b0;
    expand_and_evaluate(s, &b, 0);
    Node *root = &s->nodes[s->root];

    if (policy_out) {
        for (int i = 0; i <= s->nn; i++) policy_out[i] = 0.0f;
    }
    if (root->terminal || root->n_edges == 0) {
        if (policy_out) policy_out[s->nn] = 1.0f;   /* pass */
        if (root_value_out) *root_value_out = root->terminal_value;
        return M_PASS;
    }
    if (noise_eps > 0.0f && dirichlet_alpha > 0.0f) add_dirichlet_noise(s, dirichlet_alpha, noise_eps);

    for (int i = 0; i < sims; i++) {
        s->nodes[s->root].visits++;
        simulate(s, &b, 0);
        if (s->progress_done) *s->progress_done = i + 1;
    }
    root = &s->nodes[s->root];                 /* simulate() may have reallocated the pool */

    int   total_visits = 0;
    float best_visits = -1.0f;
    int   best_move = s->edges[root->first_edge].move;
    for (int i = 0; i < root->n_edges; i++) {
        const Edge *e = &s->edges[root->first_edge + i];
        total_visits += e->visits;
        if ((float)e->visits > best_visits) { best_visits = (float)e->visits; best_move = e->move; }
    }

    /* The policy target is always the normalised visit distribution pi; the
       temperature only affects which move is actually played. */
    if (policy_out) {
        double sum = 0.0;
        for (int i = 0; i <= s->nn; i++) policy_out[i] = 0.0f;
        for (int i = 0; i < root->n_edges; i++) {
            const Edge *e = &s->edges[root->first_edge + i];
            policy_out[board_policy_index(&b, e->move)] = (float)e->visits;
            sum += (double)e->visits;
        }
        if (sum > 0) {
            for (int i = 0; i <= s->nn; i++) policy_out[i] = (float)((double)policy_out[i] / sum);
        } else {
            policy_out[board_policy_index(&b, best_move)] = 1.0f;
        }
    }
    if (temperature > 1e-3f && policy_out) {
        const double r = rng_double(&s->rng);
        double acc = 0.0;
        for (int i = 0; i <= s->nn; i++) {
            acc += (double)policy_out[i];
            if (r < acc) return board_move_from_index(&b, i);
        }
    }
    if (root_value_out) {
        double acc = 0.0;
        for (int i = 0; i < root->n_edges; i++) {
            const Edge *e = &s->edges[root->first_edge + i];
            acc += -(double)e->value_sum;          /* edge values are the opponent's view */
        }
        *root_value_out = total_visits > 0 ? (float)(acc / (double)total_visits) : 0.0f;
    }
    return best_move;
}

void search_root_visits(const Search *s, int *out_counts) {
    const Node *root = &s->nodes[s->root];
    for (int i = 0; i <= s->nn; i++) out_counts[i] = 0;
    if (root->n_edges <= 0) return;
    Board b;
    board_init(&b, s->size);
    for (int i = 0; i < root->n_edges; i++) {
        const Edge *e = &s->edges[root->first_edge + i];
        out_counts[board_policy_index(&b, e->move)] = e->visits;
    }
}
