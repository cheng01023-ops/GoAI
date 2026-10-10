#include "board.h"
#include "test_util.h"
#include <string.h>
#include <math.h>

static int P(int size, int x, int y) { return y * size + x; }

static void put(Board *b, int x, int y, int color) { b->cell[P(b->size, x, y)] = (int8_t)color; }

int test_board_run(void) {
    /* --- basic setup --- */
    Board b;
    board_init(&b, 9);
    CHECK(b.size == 9 && b.to_move == 1 && b.ko == -1, "fresh board");
    CHECK(board_legal_moves(&b, (int[100]){0}, true) == 82, "empty 9x9 has 81 moves + pass");

    /* --- capture of a single stone --- */
    board_init(&b, 9);
    put(&b, 4, 4, 2);
    put(&b, 3, 4, 1); put(&b, 5, 4, 1); put(&b, 4, 3, 1);
    b.to_move = 1; b.nmoves = 4;
    CHECK(board_play(&b, P(9, 4, 5)) == true, "capture move is legal");
    CHECK(b.cell[P(9, 4, 4)] == 0, "captured stone removed");
    CHECK(b.captures[1] == 1, "black prisoner count == 1");
    CHECK(b.cell[P(9, 4, 5)] == 1, "capturing stone stays on board");

    /* --- suicide is illegal and leaves the board untouched --- */
    board_init(&b, 9);
    put(&b, 3, 4, 2); put(&b, 5, 4, 2); put(&b, 4, 3, 2); put(&b, 4, 5, 2);
    b.to_move = 1; b.nmoves = 4;
    {
        uint64_t h = b.hash; int8_t c0 = b.cell[P(9, 4, 4)];
        CHECK(board_play(&b, P(9, 4, 4)) == false, "suicide rejected");
        CHECK(b.hash == h && b.cell[P(9, 4, 4)] == c0 && b.to_move == 1, "board unchanged after suicide");
    }

    /* --- multi-stone capture (whole group) --- */
    board_init(&b, 9);
    put(&b, 3, 3, 2); put(&b, 4, 3, 2);
    put(&b, 3, 2, 1); put(&b, 4, 2, 1); put(&b, 2, 3, 1); put(&b, 3, 4, 1); put(&b, 4, 4, 1);
    b.to_move = 1; b.nmoves = 7;
    CHECK(board_play(&b, P(9, 5, 3)) == true, "filling last liberty is legal");
    CHECK(b.captures[1] == 2, "two stones captured at once");
    CHECK(b.cell[P(9, 3, 3)] == 0 && b.cell[P(9, 4, 3)] == 0, "group removed");

    /* --- simple ko --- */
    board_init(&b, 9);
    /* classic ko shape:
         . X O .
         X O . O
         . X O .   -> black captures at (5,4), the ko point becomes (4,4) */
    put(&b, 4, 3, 1); put(&b, 5, 3, 2);
    put(&b, 3, 4, 1); put(&b, 4, 4, 2); put(&b, 6, 4, 2);
    put(&b, 4, 5, 1); put(&b, 5, 5, 2);
    b.to_move = 1; b.nmoves = 7;
    CHECK(board_play(&b, P(9, 5, 4)) == true, "ko capture legal");
    CHECK(b.captures[1] == 1, "ko capture takes one stone");
    CHECK(b.ko == P(9, 4, 4), "ko point set to recapture spot");
    CHECK(board_is_legal_move(&b, P(9, 4, 4)) == false, "immediate recapture forbidden");
    CHECK(board_play(&b, P(9, 4, 4)) == false, "immediate recapture rejected");
    CHECK(board_play(&b, P(9, 0, 0)) == true, "playing elsewhere lifts the ko");
    CHECK(b.ko == -1 && board_is_legal_move(&b, P(9, 4, 4)) == true, "recapture allowed after a move elsewhere");

    /* --- scoring: single black stone owns the whole empty board --- */
    board_init(&b, 9);
    put(&b, 4, 4, 1);
    CHECK(fabs(board_score(&b, 0.0) - 81.0) < 1e-9, "one black stone => 81 points (got %.1f)", board_score(&b, 0.0));
    CHECK(fabs(board_score(&b, 7.0) - 74.0) < 1e-9, "komi subtracted");
    CHECK(board_winner(&b, 7.0) == 1, "black wins by 74");

    /* --- scoring: both colours present, empty region is neutral --- */
    board_init(&b, 9);
    put(&b, 4, 4, 1); put(&b, 0, 0, 2);
    CHECK(fabs(board_score(&b, 0.0)) < 1e-9, "shared empty region is dame");
    CHECK(board_winner(&b, 7.0) == 2, "white wins with komi");

    /* --- eye-like detection --- */
    board_init(&b, 9);
    put(&b, 3, 4, 1); put(&b, 5, 4, 1); put(&b, 4, 3, 1); put(&b, 4, 5, 1);
    CHECK(board_point_is_eye_like(&b, P(9, 4, 4), 1) == true, "single point eye detected");
    CHECK(board_point_is_eye_like(&b, P(9, 4, 4), 2) == false, "not an eye for white");

    /* --- 终局归属（第 ⑤ 步的领地标签）--- */
    {
        int8_t own[BOARD_MAX_POINTS];
        /* 空盘：没有棋子 -> 全部中立 */
        board_init(&b, 9);
        board_ownership(&b, own);
        {
            int nz = 0;
            for (int p = 0; p < 81; p++) if (own[p] != 0) nz++;
            CHECK(nz == 0, "empty board has no ownership (%d non-zero)", nz);
        }
        /* 一颗黑子独占全盘（与 board_score 一致） */
        board_init(&b, 9);
        put(&b, 4, 4, 1);
        board_ownership(&b, own);
        {
            int nb = 0, nw = 0, nn = 0;
            for (int p = 0; p < 81; p++) { if (own[p] > 0) nb++; else if (own[p] < 0) nw++; else nn++; }
            CHECK(nb == 81 && nw == 0, "one black stone owns all 81 points (b=%d w=%d)", nb, nw);
            CHECK(own[P(9, 4, 4)] == 1, "the black stone itself is black-owned");
            (void)nn;
        }
        /* 黑白各一颗：空区同时接触两色 -> 中立；两颗子归各自颜色 */
        board_init(&b, 9);
        put(&b, 4, 4, 1); put(&b, 0, 0, 2);
        board_ownership(&b, own);
        {
            int nb = 0, nw = 0, nneutral = 0;
            for (int p = 0; p < 81; p++) { if (own[p] > 0) nb++; else if (own[p] < 0) nw++; else nneutral++; }
            CHECK(nb == 1 && nw == 1, "only the two stones are owned (b=%d w=%d)", nb, nw);
            CHECK(nneutral == 79, "the shared empty region is neutral (%d)", nneutral);
            CHECK(own[P(9, 0, 0)] == -1, "the white stone is white-owned");
        }
        /* 两块被围住的地：左边黑、右边白、中间接触双方 -> 中立
             X . . . O        X = 黑墙（第 2 列），O = 白墙（第 6 列） */
        board_init(&b, 9);
        for (int y = 0; y < 9; y++) { put(&b, 2, y, 1); put(&b, 6, y, 2); }
        board_ownership(&b, own);
        CHECK(own[P(9, 0, 4)] == 1, "empty column left of the black wall is black-owned");
        CHECK(own[P(9, 8, 4)] == -1, "empty column right of the white wall is white-owned");
        CHECK(own[P(9, 4, 4)] == 0, "the gap between the two walls is neutral");
        CHECK(own[P(9, 2, 4)] == 1 && own[P(9, 6, 4)] == -1, "wall stones belong to their colour");
        /* 与 Tromp-Taylor 数子结果一致：黑地 - 白地 == board_score(komi=0) */
        {
            int diff = 0;
            for (int p = 0; p < 81; p++) diff += own[p];
            CHECK(fabs((double)diff - board_score(&b, 0.0)) < 1e-9,
                  "ownership sum matches area score (%d vs %.1f)", diff, board_score(&b, 0.0));
        }
        /* 随机对局（下满盘面）后也必须一致 */
        {
            Rng rng; rng_seed(&rng, 777);
            Board g; board_init(&g, 9);
            for (int i = 0; i < 200 && g.passes < 2; i++) {
                const int mv = board_random_move(&g, &rng, 0);
                if (!board_play(&g, mv)) board_play(&g, M_PASS);
            }
            board_ownership(&g, own);
            int diff = 0, bad = 0;
            for (int p = 0; p < 81; p++) {
                diff += own[p];
                if (own[p] < -1 || own[p] > 1) bad++;
            }
            CHECK(bad == 0, "ownership labels stay in {-1,0,+1}");
            CHECK(fabs((double)diff - board_score(&g, 0.0)) < 1e-9,
                  "ownership sum matches area score on a played-out board (%d vs %.1f)",
                  diff, board_score(&g, 0.0));
        }
    }

    /* --- playouts always finish and return a valid result --- */
    {
        Rng rng; rng_seed(&rng, 12345);
        int bad = 0;
        for (int g = 0; g < 8; g++) {
            board_init(&b, 9);
            float v = board_playout_value(&b, &rng, 400, 7.0);
            if (!(v == 1.0f || v == -1.0f || v == 0.0f)) bad++;
            if (b.passes < 1 && b.nmoves < 400) bad++;
        }
        CHECK(bad == 0, "all playouts terminated with a valid result");
    }

    /* --- zobrist hashing is stable & order independent --- */
    {
        Board a, c;
        board_init(&a, 9); board_init(&c, 9);
        board_play(&a, P(9, 2, 2)); board_play(&a, P(9, 6, 6));
        board_play(&a, P(9, 2, 6)); board_play(&a, P(9, 6, 2));
        board_play(&c, P(9, 2, 6)); board_play(&c, P(9, 6, 2));
        board_play(&c, P(9, 2, 2)); board_play(&c, P(9, 6, 6));
        CHECK(a.hash == c.hash, "hash independent of move order");
    }

    /* --- vertex parsing round trip --- */
    {
        char s[16];
        board_init(&b, 9);
        board_vertex(&b, P(9, 4, 4), s, sizeof s);
        CHECK(strcmp(s, "E5") == 0, "vertex formatting (got %s)", s);
        CHECK(board_parse_vertex(&b, "E5") == P(9, 4, 4), "vertex parsing");
        CHECK(board_parse_vertex(&b, "J9") == P(9, 8, 0), "column J skips I");
        CHECK(board_parse_vertex(&b, "pass") == M_PASS, "pass parsing");
        CHECK(board_parse_vertex(&b, "Z9") == M_NONE, "out of range rejected");
    }
    return 0;
}
