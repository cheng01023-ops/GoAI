/* board.h - Go board, rules and scoring (Tromp-Taylor area scoring) */
#ifndef GOAI_BOARD_H
#define GOAI_BOARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "rand.h"

#define BOARD_MAX 19
#define BOARD_MAX_POINTS (BOARD_MAX * BOARD_MAX)

/* Move encoding: 0 .. size*size-1 are board points (p = y*size + x, y = 0 is the TOP row).
   M_PASS is a pass, M_NONE means "no move". */
#define M_NONE (-1)
#define M_PASS (-2)

typedef struct {
    int      size;                    /* 9, 13 or 19 */
    int      to_move;                 /* 1 = black, 2 = white */
    int      ko;                      /* forbidden point (simple ko) or -1 */
    int      passes;                  /* consecutive passes */
    int      nmoves;                  /* moves played in the game so far */
    int      captures[3];             /* prisoners taken by colour 1 / 2 */
    int8_t   cell[BOARD_MAX_POINTS];  /* 0 empty, 1 black, 2 white */
    uint64_t hash;                    /* zobrist hash */
} Board;

void     board_init(Board *b, int size);
void     board_copy(Board *dst, const Board *src);
int      board_neighbors(const Board *b, int p, int *out);   /* returns count */
int      board_point_count(const Board *b);
bool     board_play(Board *b, int move);                     /* false if illegal; board untouched */
bool     board_is_legal_move(const Board *b, int move);
int      board_legal_moves(const Board *b, int *out, bool allow_pass);
int      board_policy_index(const Board *b, int move);       /* index in [0, nn] ; pass -> nn */
int      board_move_from_index(const Board *b, int idx);
bool     board_point_is_eye_like(const Board *b, int p, int color);
int      board_random_move(const Board *b, Rng *rng, int min_moves_before_pass);
double   board_score(const Board *b, double komi);           /* black - white - komi */
/* 终局归属（黑方视角，第 ⑤ 步的领地标签）：
   +1 = 黑确定拥有（黑子 / 只被黑子围住的空区），-1 = 白确定拥有，
    0 = 中立（空区同时接触两色，或者空盘没有棋子）。
   与 board_score 的 Tromp-Taylor 数子规则一致（棋盘上所有子都算活）。 */
void     board_ownership(const Board *b, int8_t *own);
int      board_winner(const Board *b, double komi);          /* 1 black, 2 white, 0 jigo */
float    board_playout_value(Board *b, Rng *rng, int max_moves, double komi);
void     board_to_string(const Board *b, char *buf, size_t buflen);
void     board_vertex(const Board *b, int move, char *out, size_t n); /* GTP vertex e.g. "E5" */
int      board_parse_vertex(const Board *b, const char *s);           /* -> move or M_NONE */
uint64_t board_zobrist(const Board *b);

#endif /* GOAI_BOARD_H */
