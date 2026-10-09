/* board.c - Go board, rules and Tromp-Taylor scoring */
#include "board.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static uint64_t ZOB[2][BOARD_MAX_POINTS];
static int      zob_ready = 0;

static void zob_init(void) {
    uint64_t x = 0xC0FFEE123456789ULL;
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < BOARD_MAX_POINTS; i++)
            ZOB[c][i] = rng_splitmix64(&x);
    zob_ready = 1;
}

int board_point_count(const Board *b) { return b->size * b->size; }

void board_init(Board *b, int size) {
    if (!zob_ready) zob_init();
    memset(b, 0, sizeof(*b));
    b->size = size;
    b->to_move = 1;
    b->ko = -1;
    b->hash = 0;
}

void board_copy(Board *dst, const Board *src) { memcpy(dst, src, sizeof(*dst)); }

int board_neighbors(const Board *b, int p, int *out) {
    const int n = b->size, x = p % n, y = p / n;
    int k = 0;
    if (y > 0)     out[k++] = p - n;
    if (y < n - 1) out[k++] = p + n;
    if (x > 0)     out[k++] = p - 1;
    if (x < n - 1) out[k++] = p + 1;
    return k;
}

typedef struct {
    int stones[BOARD_MAX_POINTS];
    int nstones;
    int libs[BOARD_MAX_POINTS];
    int nlibs;
} Group;

static void group_flood(const Board *b, int start, Group *g) {
    const int npoints = b->size * b->size;
    int8_t seen[BOARD_MAX_POINTS];
    int    stack[BOARD_MAX_POINTS];
    memset(seen, 0, (size_t)npoints);
    const int8_t color = b->cell[start];
    int sp = 0;
    g->nstones = 0;
    g->nlibs = 0;
    stack[sp++] = start;
    seen[start] = 1;
    while (sp > 0) {
        const int p = stack[--sp];
        g->stones[g->nstones++] = p;
        int nb[4];
        const int c = board_neighbors(b, p, nb);
        for (int i = 0; i < c; i++) {
            const int q = nb[i];
            const int8_t v = b->cell[q];
            if (v == 0) {
                if (seen[q] == 0) { seen[q] = 2; g->libs[g->nlibs++] = q; }
            } else if (v == color && seen[q] == 0) {
                seen[q] = 1;
                stack[sp++] = q;
            }
        }
    }
}

/* Quick legality test for playing at empty point p (no board copy needed). */
static bool point_playable_quick(const Board *b, int p) {
    if (b->cell[p] != 0 || p == b->ko) return false;
    int nb[4];
    const int c = board_neighbors(b, p, nb);
    const int color = b->to_move;
    for (int i = 0; i < c; i++)
        if (b->cell[nb[i]] == 0) return true;             /* new group gets a liberty */
    for (int i = 0; i < c; i++) {
        if (b->cell[nb[i]] == color) {
            Group g;
            group_flood(b, nb[i], &g);
            for (int k = 0; k < g.nlibs; k++)
                if (g.libs[k] != p) return true;          /* own group keeps a liberty */
        }
    }
    for (int i = 0; i < c; i++) {
        const int8_t v = b->cell[nb[i]];
        if (v != 0 && v != color) {
            Group g;
            group_flood(b, nb[i], &g);
            if (g.nlibs == 1) return true;                /* capture -> legal */
        }
    }
    return false;                                          /* suicide */
}

/* Number of liberties the group played at p would have (used to avoid
   self-atari in playouts). Shared liberties may be counted twice, which is a
   safe over-estimate for this heuristic. */
static int play_liberties(const Board *b, int p) {
    int nb[4];
    const int c = board_neighbors(b, p, nb);
    const int color = b->to_move, opp = 3 - color;
    int libs = 0;
    for (int i = 0; i < c; i++) if (b->cell[nb[i]] == 0) libs++;
    for (int i = 0; i < c; i++) {
        if (b->cell[nb[i]] == color) {
            Group g;
            group_flood(b, nb[i], &g);
            for (int k = 0; k < g.nlibs; k++) if (g.libs[k] != p) libs++;
        }
    }
    for (int i = 0; i < c; i++) {
        if (b->cell[nb[i]] == opp) {
            Group g;
            group_flood(b, nb[i], &g);
            if (g.nlibs == 1) libs += g.nstones;      /* capturing is always safe */
        }
    }
    return libs;
}

bool board_play(Board *b, int move) {
    if (move == M_PASS) {
        b->passes++;
        b->ko = -1;
        b->to_move = 3 - b->to_move;
        b->nmoves++;
        return true;
    }
    const int npoints = b->size * b->size;
    if (move < 0 || move >= npoints) return false;
    if (b->cell[move] != 0) return false;
    if (move == b->ko) return false;

    int8_t   saved_cell[BOARD_MAX_POINTS];
    memcpy(saved_cell, b->cell, (size_t)npoints);
    const uint64_t saved_hash = b->hash;
    const int      saved_ko = b->ko;
    const int      saved_caps = b->captures[b->to_move];

    const int color = b->to_move, opp = 3 - color;
    b->cell[move] = (int8_t)color;
    b->hash ^= ZOB[color - 1][move];

    int captured = 0;
    int nb[4];
    const int c = board_neighbors(b, move, nb);
    for (int i = 0; i < c; i++) {
        const int q = nb[i];
        if (b->cell[q] != opp) continue;
        Group g;
        group_flood(b, q, &g);
        if (g.nlibs != 0) continue;
        for (int k = 0; k < g.nstones; k++) {
            const int s = g.stones[k];
            b->cell[s] = 0;
            b->hash ^= ZOB[opp - 1][s];
        }
        captured += g.nstones;
    }

    Group mine;
    group_flood(b, move, &mine);
    if (mine.nlibs == 0) {                       /* suicide: take it back */
        memcpy(b->cell, saved_cell, (size_t)npoints);
        b->hash = saved_hash;
        b->ko = saved_ko;
        b->captures[color] = saved_caps;
        return false;
    }

    b->ko = -1;
    if (captured == 1 && mine.nstones == 1 && mine.nlibs == 1)
        b->ko = mine.libs[0];                    /* simple ko */

    b->captures[color] += captured;
    b->passes = 0;
    b->to_move = opp;
    b->nmoves++;
    return true;
}

bool board_is_legal_move(const Board *b, int move) {
    if (move == M_PASS) return true;
    const int npoints = b->size * b->size;
    if (move < 0 || move >= npoints) return false;
    return point_playable_quick(b, move);
}

int board_legal_moves(const Board *b, int *out, bool allow_pass) {
    const int npoints = b->size * b->size;
    int k = 0;
    for (int p = 0; p < npoints; p++) {
        if (b->cell[p] != 0) continue;
        if (point_playable_quick(b, p)) out[k++] = p;
    }
    if (allow_pass) out[k++] = M_PASS;
    return k;
}

int board_policy_index(const Board *b, int move) {
    if (move == M_PASS) return b->size * b->size;
    return move;
}

int board_move_from_index(const Board *b, int idx) {
    const int nn = b->size * b->size;
    if (idx == nn) return M_PASS;
    if (idx < 0 || idx > nn) return M_NONE;
    return idx;
}

bool board_point_is_eye_like(const Board *b, int p, int color) {
    if (b->cell[p] != 0) return false;
    const int n = b->size, x = p % n, y = p / n;
    int nb[4];
    const int c = board_neighbors(b, p, nb);
    for (int i = 0; i < c; i++)
        if (b->cell[nb[i]] != color) return false;
    if (x == 0 || y == 0 || x == n - 1 || y == n - 1) return true;
    const int opp = 3 - color;
    int opp_diag = 0;
    const int dx[4] = { -1, 1, -1, 1 }, dy[4] = { -1, -1, 1, 1 };
    for (int i = 0; i < 4; i++) {
        const int q = (y + dy[i]) * n + (x + dx[i]);
        if (b->cell[q] == opp) opp_diag++;
    }
    return opp_diag <= 1;                        /* 2+ opponent corners => false eye */
}

int board_random_move(const Board *b, Rng *rng, int min_moves_before_pass) {
    const int npoints = b->size * b->size;
    int cand[BOARD_MAX_POINTS + 1], safe[BOARD_MAX_POINTS + 1];
    int nc = 0, ns = 0;
    for (int p = 0; p < npoints; p++) {
        if (b->cell[p] != 0) continue;
        if (board_point_is_eye_like(b, p, b->to_move)) continue;
        if (!point_playable_quick(b, p)) continue;
        cand[nc++] = p;
        if (play_liberties(b, p) >= 2) safe[ns++] = p;
    }
    if (ns > 0) { memcpy(cand, safe, (size_t)ns * sizeof(int)); nc = ns; }
    if (nc == 0) {
        for (int p = 0; p < npoints; p++) {
            if (b->cell[p] != 0) continue;
            if (!point_playable_quick(b, p)) continue;
            cand[nc++] = p;
        }
    }
    if (nc == 0) return M_PASS;
    if (b->nmoves >= min_moves_before_pass && rng_double(rng) < 0.02) return M_PASS;
    return cand[rng_below(rng, (uint32_t)nc)];
}

double board_score(const Board *b, double komi) {
    const int npoints = b->size * b->size;
    int8_t seen[BOARD_MAX_POINTS];
    memset(seen, 0, (size_t)npoints);
    int black = 0, white = 0;
    for (int p = 0; p < npoints; p++) {
        if (b->cell[p] == 1) black++;
        else if (b->cell[p] == 2) white++;
    }
    int stack[BOARD_MAX_POINTS];
    for (int start = 0; start < npoints; start++) {
        if (b->cell[start] != 0 || seen[start]) continue;
        int sp = 0, size = 0, touch_black = 0, touch_white = 0;
        stack[sp++] = start;
        seen[start] = 1;
        while (sp > 0) {
            const int p = stack[--sp];
            size++;
            int nb[4];
            const int c = board_neighbors(b, p, nb);
            for (int i = 0; i < c; i++) {
                const int q = nb[i];
                const int8_t v = b->cell[q];
                if (v == 0) {
                    if (!seen[q]) { seen[q] = 1; stack[sp++] = q; }
                } else if (v == 1) touch_black = 1;
                else touch_white = 1;
            }
        }
        if (touch_black && !touch_white) black += size;
        else if (touch_white && !touch_black) white += size;
    }
    return (double)black - (double)white - komi;
}

int board_winner(const Board *b, double komi) {
    const double s = board_score(b, komi);
    if (s > 0) return 1;
    if (s < 0) return 2;
    return 0;
}

float board_playout_value(Board *b, Rng *rng, int max_moves, double komi) {
    const int start = b->to_move;
    const int limit = b->nmoves + max_moves;
    /* Approximation used by light playouts: the game ends as soon as somebody
       passes.  Filling the whole board instead makes every group die and turns
       the score into noise, so a single pass is a much better stopping rule. */
    while (b->passes < 1 && b->nmoves < limit) {
        const int m = board_random_move(b, rng, 0);
        if (!board_play(b, m)) break;
    }
    if (b->passes < 1) {                        /* hit the move limit: score as is */
        while (b->nmoves < limit + 2) board_play(b, M_PASS);
    }
    const double s = board_score(b, komi);
    if (s > 0) return start == 1 ? 1.0f : -1.0f;
    if (s < 0) return start == 2 ? 1.0f : -1.0f;
    return 0.0f;
}

void board_to_string(const Board *b, char *buf, size_t buflen) {
    const int n = b->size;
    size_t used = 0;
    char line[128];
    int k = snprintf(line, sizeof(line), "   ");
    for (int x = 0; x < n; x++) k += snprintf(line + k, sizeof(line) - (size_t)k, "%c ", (x < 8) ? ('A' + x) : ('A' + x + 1));
    k += snprintf(line + k, sizeof(line) - (size_t)k, "\n");
    used += (size_t)snprintf(buf + used, buflen - used, "%s", line);
    for (int y = 0; y < n; y++) {
        k = snprintf(line, sizeof(line), "%2d ", n - y);
        for (int x = 0; x < n; x++) {
            const int8_t v = b->cell[y * n + x];
            k += snprintf(line + k, sizeof(line) - (size_t)k, "%c ", v == 0 ? '.' : (v == 1 ? 'X' : 'O'));
        }
        k += snprintf(line + k, sizeof(line) - (size_t)k, "%2d\n", n - y);
        used += (size_t)snprintf(buf + used, buflen - used, "%s", line);
    }
    k = snprintf(line, sizeof(line), "   ");
    for (int x = 0; x < n; x++) k += snprintf(line + k, sizeof(line) - (size_t)k, "%c ", (x < 8) ? ('A' + x) : ('A' + x + 1));
    k += snprintf(line + k, sizeof(line) - (size_t)k, "\n");
    snprintf(buf + used, buflen - used, "%s", line);
}

void board_vertex(const Board *b, int move, char *out, size_t n) {
    if (move == M_PASS) { snprintf(out, n, "pass"); return; }
    if (move < 0 || move >= b->size * b->size) { snprintf(out, n, "resign"); return; }
    const int x = move % b->size, y = move / b->size;
    const char col = (char)((x < 8) ? ('A' + x) : ('A' + x + 1));
    snprintf(out, n, "%c%d", col, b->size - y);
}

int board_parse_vertex(const Board *b, const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) return M_NONE;
    if ((s[0] == 'p' || s[0] == 'P') && (s[1] == 'a' || s[1] == 'A') && (s[2] == 's' || s[2] == 'S')) return M_PASS;
    char c = s[0];
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c < 'A' || c > 'Z') return M_NONE;
    int x = c - 'A';
    if (c > 'I') x--;
    const int n = b->size;
    if (x < 0 || x >= n) return M_NONE;
    int row = 0;
    const char *p = s + 1;
    if (*p < '0' || *p > '9') return M_NONE;
    while (*p >= '0' && *p <= '9') { row = row * 10 + (*p - '0'); p++; }
    if (row < 1 || row > n) return M_NONE;
    return (n - row) * n + x;
}

uint64_t board_zobrist(const Board *b) { return b->hash; }
