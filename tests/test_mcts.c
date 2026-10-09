#include "mcts.h"
#include "test_util.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int P(int size, int x, int y) { return y * size + x; }
static void put(Board *b, int x, int y, int c) { b->cell[P(b->size, x, y)] = (int8_t)c; }

int test_mcts_run(void) {
    /* ---- rollout mode: legal move, sane policy ---- */
    {
        Search s;
        search_init(&s, 9, NULL, 7, 7.0);
        Board b; board_init(&b, 9);
        float pol[82], rv = 0;
        const int m = search_run(&s, &b, 200, 0.0f, 0.0f, 0.0f, pol, &rv);
        CHECK(m >= 0 && m < 81, "rollout search returns an on-board move");
        double sum = 0; for (int i = 0; i < 82; i++) sum += pol[i];
        CHECK(fabs(sum - 1.0) < 1e-4, "policy normalised (%.4f)", sum);
        CHECK(pol[P(9, m % 9, m / 9)] > 0.0f, "chosen move has non-zero probability");
        CHECK(rv >= -1.0f && rv <= 1.0f, "root value in range");
        search_free(&s);
    }

    /* ---- terminal handling and value sign: a full board won by black ---- */
    {
        Search s;
        search_init(&s, 9, NULL, 11, 7.0);
        Board b; board_init(&b, 9);
        for (int p = 0; p < 81; p++) b.cell[p] = 1;   /* every point black, komi 7 */
        b.to_move = 1; b.passes = 1; b.nmoves = 100;
        float pol[82], rv = 0;
        const int m = search_run(&s, &b, 100, 0.0f, 0.0f, 0.0f, pol, &rv);
        CHECK(m == M_PASS, "board full => only pass is legal");
        CHECK(pol[81] > 0.99f, "policy is a pass");
        CHECK(rv > 0.5f, "black wins 81-0-7 => root value positive (%.3f)", rv);
        search_free(&s);
    }
    {
        Search s;
        search_init(&s, 9, NULL, 12, 7.0);
        Board b; board_init(&b, 9);
        for (int p = 0; p < 81; p++) b.cell[p] = 2;   /* every point white */
        b.to_move = 1; b.passes = 1; b.nmoves = 100;
        float pol[82], rv = 0;
        const int m = search_run(&s, &b, 100, 0.0f, 0.0f, 0.0f, pol, &rv);
        CHECK(m == M_PASS, "black has no move but pass");
        CHECK(rv < -0.5f, "black is 81 points behind => root value negative (%.3f)", rv);
        search_free(&s);
    }

    /* ---- value sign on decided boards (rollout values are now sane) ---- */
    {
        Search s;
        search_init(&s, 9, NULL, 13, 7.0);
        Board b; board_init(&b, 9);
        for (int x = 0; x < 4; x++) for (int y = 0; y < 9; y++) put(&b, x, y, 1);
        b.to_move = 1; b.nmoves = 36;
        float pol[82], rv = 0;
        (void)search_run(&s, &b, 150, 0.0f, 0.0f, 0.0f, pol, &rv);
        CHECK(rv > 0.5f, "black owns half the board => positive value (%.3f)", rv);
        search_free(&s);
    }
    {
        Search s;
        search_init(&s, 9, NULL, 14, 7.0);
        Board b; board_init(&b, 9);
        for (int x = 0; x < 4; x++) for (int y = 0; y < 9; y++) put(&b, x, y, 2);
        b.to_move = 1; b.nmoves = 36;
        float pol[82], rv = 0;
        (void)search_run(&s, &b, 150, 0.0f, 0.0f, 0.0f, pol, &rv);
        CHECK(rv < -0.5f, "white owns half the board => negative value (%.3f)", rv);
        search_free(&s);
    }

    /* ---- net mode ---- */
    {
        Net net; net_init(&net, 9, NET_FEATURE_PLANES, 8, 16, 5);
        Search s; search_init(&s, 9, &net, 3, 7.0);
        Board b; board_init(&b, 9);
        board_play(&b, P(9, 4, 4));
        float pol[82], rv = 0;
        const int m = search_run(&s, &b, 50, 0.3f, 0.25f, 1.0f, pol, &rv);
        CHECK(m >= 0 && m < 81, "net search returns an on-board move");
        CHECK(board_is_legal_move(&b, m), "net search move is legal");
        double sum = 0; for (int i = 0; i < 82; i++) sum += pol[i];
        CHECK(fabs(sum - 1.0) < 1e-4, "net policy normalised");
        CHECK(rv >= -1.0f && rv <= 1.0f, "net root value in range");
        search_free(&s); net_free(&net);
    }

    /* ---- end to end: a network trained on one position steers the search ---- */
    {
        const int C = 8, H = 16;
        Net net; NetCache cache;
        net_init(&net, 9, NET_FEATURE_PLANES, C, H, 21);
        net_cache_init(&net, &cache);
        Board b; board_init(&b, 9);
        board_play(&b, P(9, 4, 4)); board_play(&b, P(9, 2, 2)); board_play(&b, P(9, 6, 6));
        const int target = P(9, 4, 2);
        float *x = (float *)calloc((size_t)NET_FEATURE_PLANES * 81, sizeof(float));
        float *pi = (float *)calloc(82, sizeof(float));
        net_features(&b, x);
        pi[target] = 1.0f;
        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        float *adam_m = (float *)calloc((size_t)net.n_params, sizeof(float));
        float *adam_v = (float *)calloc((size_t)net.n_params, sizeof(float));
        float pol[82], val;
        for (int step = 1; step <= 400; step++) {
            net_zero_grad(&net, grad);
            net_forward(&net, &cache, x, pol, &val);
            net_backward(&net, &cache, pi, 1.0f, grad, NULL, NULL);
            net_adam(&net, grad, adam_m, adam_v, step, 0.02f, 0.0f, 5.0f);
        }
        Search s; search_init(&s, 9, &net, 9, 7.0);
        float p2[82], rv = 0;
        const int m = search_run(&s, &b, 100, 0.0f, 0.0f, 0.0f, p2, &rv);
        CHECK(m == target, "trained network steers the search to its move (got %d,%d want %d,%d)",
              m % 9, m / 9, target % 9, target / 9);
        search_free(&s);
        free(x); free(pi); free(grad); free(adam_m); free(adam_v);
        net_cache_free(&cache); net_free(&net);
    }

    /* ---- tree grows / memory is reusable across many searches ---- */
    {
        Net net; net_init(&net, 9, NET_FEATURE_PLANES, 8, 16, 6);
        Search s; search_init(&s, 9, &net, 4, 7.0);
        Board b; board_init(&b, 9);
        int bad = 0;
        for (int k = 0; k < 20; k++) {
            float pol[82], rv;
            const int m = search_run(&s, &b, 40, 0.3f, 0.0f, 0.0f, pol, &rv);
            if (m < 0 || m >= 81 || !board_is_legal_move(&b, m)) bad++;
            if (!board_play(&b, m)) bad++;
            if (b.passes >= 2) break;
        }
        CHECK(bad == 0, "20 chained searches stayed legal");
        CHECK(s.n_nodes > 1, "tree actually grew (%d nodes)", s.n_nodes);
        search_free(&s); net_free(&net);
    }
    /* 7) 树复用：走完一手后根前进到子树，下次搜索应继承之前的访问 */
    {
        Net net;
        net_init(&net, 9, 4, 16, 32, 12345);
        NetCache cache;
        net_cache_init(&net, &cache);
        Search s;
        search_init(&s, 9, &net, 999, 7.0);
        Board b;
        board_init(&b, 9);
        float pol[82], val = 0;
        const int mv = search_run(&s, &b, 120, 0.0f, 0.0f, 0.0f, pol, &val);
        CHECK(s.root == 0, "首次搜索根节点应为 0");
        const int visits_before = s.nodes[0].visits;
        CHECK(s.nodes[0].expanded, "根节点应已展开");
        CHECK(mv >= 0 && mv < 81, "应返回合法着法（%d）", mv);
        /* 走这一手，标记复用 */
        CHECK(board_is_legal_move(&b, mv), "着法合法");
        board_play(&b, mv);
        s.pending_advance = mv;
        float pol2[82];
        const int mv2 = search_run(&s, &b, 60, 0.0f, 0.0f, 0.0f, pol2, &val);
        CHECK(s.root != 0, "复用后根节点下标应不再是 0（实际 %d）", s.root);
        CHECK(s.nodes[s.root].visits >= 60, "复用后根节点访问数应至少包含本轮 60 次（%d）", s.nodes[s.root].visits);
        CHECK(mv2 >= 0 && mv2 < 82, "复用后仍返回合法着法（%d）", mv2);
        /* 不复用时应该重建 */
        s.pending_advance = -1;
        search_run(&s, &b, 30, 0.0f, 0.0f, 0.0f, pol2, &val);
        CHECK(s.root == 0, "未标记复用时根节点应重置为 0");
        (void)visits_before;
        search_free(&s);
        net_cache_free(&cache);
        net_free(&net);
    }
    return 0;
}
