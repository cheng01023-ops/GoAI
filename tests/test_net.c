#include "net.h"
#include "test_util.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* loss of a single example, gradients written into grad (or discarded) */
static float example_loss(Net *net, NetCache *c, const float *x, const float *pi, float z, float *grad) {
    float pol[BOARD_MAX_POINTS + 1], v;
    float pl = 0, vl = 0;
    net_forward(net, c, x, pol, &v);
    return net_backward(net, c, pi, z, grad, &pl, &vl);
}

int test_net_run(void) {
    /* ---------- shapes / softmax / value range ---------- */
    {
        Net net; NetCache c;
        net_init(&net, 9, NET_FEATURE_PLANES, 8, 16, 7);
        net_cache_init(&net, &c);
        float x[4 * 81], pol[82], v;
        Board b; board_init(&b, 9);
        net_features(&b, x);
        net_forward(&net, &c, x, pol, &v);
        double s = 0; for (int i = 0; i < 82; i++) s += pol[i];
        CHECK(fabs(s - 1.0) < 1e-5, "policy sums to 1 (got %.6f)", s);
        CHECK(v > -1.0f && v < 1.0f, "value in (-1,1)");
        CHECK(fabs(x[3 * 81 + 40] - 1.0f) < 1e-6, "ones plane filled");
        net_cache_free(&c); net_free(&net);
    }

    /* ---------- numeric gradient check (the important one) ---------- */
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Net net; NetCache c;
        net_init(&net, size, P, C, H, 42);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;

        Rng rng; rng_seed(&rng, 2024);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = rng_double(&rng) < 0.4 ? 1.0f : 0.0f;
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);
        const float z = 1.0f;

        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        net_zero_grad(&net, grad);
        example_loss(&net, &c, x, pi, z, grad);

        const float h = 1e-2f;
        int    checked = 0, bad = 0;
        double sum_an = 0, sum_num = 0, sum_a2 = 0, sum_n2 = 0, sum_an2 = 0;
        for (int t = 0; t < 80; t++) {
            const int i = (int)rng_below(&rng, (uint32_t)net.n_params);
            const float w0 = net.params[i];
            net.params[i] = w0 + h;
            const float lp = example_loss(&net, &c, x, pi, z, NULL);
            net.params[i] = w0 - h;
            const float lm = example_loss(&net, &c, x, pi, z, NULL);
            net.params[i] = w0;
            const double num = (double)(lp - lm) / (2.0 * h);
            const double an = grad[i];
            checked++;
            const double tol = 2e-3 + 0.05 * fabs(an);
            if (fabs(num - an) > tol) bad++;
            sum_an += an; sum_num += num; sum_a2 += an * an; sum_n2 += num * num; sum_an2 += an * num;
        }
        const double denom = sqrt((sum_a2 - sum_an * sum_an / checked) * (sum_n2 - sum_num * sum_num / checked));
        const double corr = denom > 0 ? (sum_an2 - sum_an * sum_num / checked) / denom : 0.0;
        CHECK(bad == 0, "weight gradients match finite differences (%d/%d outside tolerance)", bad, checked);
        CHECK(corr > 0.99, "gradient correlation with finite differences = %.5f", corr);

        free(grad);
        net_cache_free(&c); net_free(&net);
        free(x); free(pi);
    }

    /* ---------- input gradient check ---------- */
    {
        const int size = 4, P = NET_FEATURE_PLANES, C = 4, H = 6;
        Net net; NetCache c;
        net_init(&net, size, P, C, H, 99);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;
        Rng rng; rng_seed(&rng, 555);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = (float)rng_double(&rng);
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);
        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        net_zero_grad(&net, grad);
        float pol[BOARD_MAX_POINTS + 1], v;
        net_forward(&net, &c, x, pol, &v);
        net_backward(&net, &c, pi, -1.0f, grad, NULL, NULL);
        const float h = 1e-2f;
        int bad = 0, checked = 0;
        for (int t = 0; t < 40; t++) {
            const int i = (int)rng_below(&rng, (uint32_t)(P * nn));
            const float x0 = x[i];
            x[i] = x0 + h; const float lp = example_loss(&net, &c, x, pi, -1.0f, NULL);
            x[i] = x0 - h; const float lm = example_loss(&net, &c, x, pi, -1.0f, NULL);
            x[i] = x0;
            const double num = (double)(lp - lm) / (2.0 * h);
            const double an = c.dx[i];
            checked++;
            if (fabs(num - an) > 2e-3 + 0.05 * fabs(an)) bad++;
        }
        CHECK(bad == 0, "input gradients match finite differences (%d/%d bad)", bad, checked);
        free(grad); free(x); free(pi);
        net_cache_free(&c); net_free(&net);
    }

    /* ---------- save / load round trip ---------- */
    {
        Net a, b;
        net_init(&a, 9, NET_FEATURE_PLANES, 8, 16, 3);
        Rng rng; rng_seed(&rng, 1);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rng) * 0.01);
        CHECK(net_save(&a, "/tmp/goai_net_test.bin"), "net_save");
        net_init(&b, 9, NET_FEATURE_PLANES, 8, 16, 4);
        CHECK(net_load(&b, "/tmp/goai_net_test.bin"), "net_load");
        CHECK(b.n_params == a.n_params, "param count preserved");
        int diff = 0;
        for (int i = 0; i < a.n_params; i++) if (a.params[i] != b.params[i]) diff++;
        CHECK(diff == 0, "weights identical after round trip (%d diffs)", diff);
        net_free(&a); net_free(&b);
    }

    /* ---------- training actually reduces the loss (overfit one example) ---------- */
    {
        const int size = 9, P = NET_FEATURE_PLANES, C = 8, H = 16;
        Net net; NetCache c;
        net_init(&net, size, P, C, H, 11);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;
        Board bd; board_init(&bd, 9);
        board_play(&bd, 40); board_play(&bd, 30); board_play(&bd, 50);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        net_features(&bd, x);
        pi[40] = 0.7f; pi[30] = 0.2f; pi[50] = 0.1f;
        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        float *m = (float *)calloc((size_t)net.n_params, sizeof(float));
        float *vv = (float *)calloc((size_t)net.n_params, sizeof(float));
        float pol[BOARD_MAX_POINTS + 1], v;
        net_forward(&net, &c, x, pol, &v);
        const float first = example_loss(&net, &c, x, pi, 1.0f, grad);
        for (int step = 1; step <= 300; step++) {
            net_zero_grad(&net, grad);
            example_loss(&net, &c, x, pi, 1.0f, grad);
            net_adam(&net, grad, m, vv, step, 0.02f, 0.0f, 5.0f);
        }
        const float last = example_loss(&net, &c, x, pi, 1.0f, grad);
        net_forward(&net, &c, x, pol, &v);
        CHECK(last < first * 0.5f, "adam reduces loss: %.4f -> %.4f", first, last);
        CHECK(pol[40] > 0.5f, "policy concentrates on the target move (p=%.3f)", pol[40]);
        CHECK(v > 0.5f, "value moves towards +1 (v=%.3f)", v);
        free(grad); free(m); free(vv); free(x); free(pi);
        net_cache_free(&c); net_free(&net);
    }
    /* ---------- 残差块的数值梯度校验（blocks = 2）---------- */
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8, BLOCKS = 2;
        Net net; NetCache c;
        net_init_ex(&net, size, P, C, H, BLOCKS, 77);
        CHECK(net.blocks == BLOCKS, "residual net has %d blocks", net.blocks);
        /* 生产代码把残差块零初始化（块初始 = 恒等映射，保护预训练权重），
           但零权重会让这些参数的梯度恒为 0、有限差分退化成噪声。
           所以这里先把块权重随机化，再校验梯度数学。 */
        {
            Rng rw;
            rng_seed(&rw, 4242);
            for (int i = 0; i < net.len_bw; i++)
                net.params[net.off_bw + i] = (float)(rng_normal(&rw) * 0.08);
        }
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;

        Rng rng; rng_seed(&rng, 555);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = rng_double(&rng) < 0.4 ? 1.0f : 0.0f;
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);
        const float z = 1.0f;

        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        net_zero_grad(&net, grad);
        example_loss(&net, &c, x, pi, z, grad);

        const float h = 3e-3f;
        int checked = 0, bad = 0;
        double worst = 0.0;
        double sum_an = 0, sum_num = 0, sum_a2 = 0, sum_n2 = 0, sum_an2 = 0;
        for (int t = 0; t < 80; t++) {
            const int i = (int)rng_below(&rng, (uint32_t)net.n_params);
            const float w0 = net.params[i];
            net.params[i] = w0 + h;
            const float lp = example_loss(&net, &c, x, pi, z, NULL);
            net.params[i] = w0 - h;
            const float lm = example_loss(&net, &c, x, pi, z, NULL);
            net.params[i] = w0;
            const double num = (double)(lp - lm) / (2.0 * h);
            const double an = grad[i];
            checked++;
            const double tol = 4e-3 + 0.08 * fabs(an);
            if (fabs(num - an) > tol) { bad++; if (fabs(num-an) > worst) worst = fabs(num-an); }
            sum_an += an; sum_num += num; sum_a2 += an * an; sum_n2 += num * num; sum_an2 += an * num;
        }
        const double denom = sqrt((sum_a2 - sum_an * sum_an / checked) * (sum_n2 - sum_num * sum_num / checked));
        const double corr = denom > 0 ? (sum_an2 - sum_an * sum_num / checked) / denom : 0.0;
        printf("    [残差块梯度] 最大绝对偏差 %.2e, 超差 %d/%d\n", worst, bad, checked);
                CHECK(bad == 0, "residual-block weight gradients match finite differences (%d/%d bad)", bad, checked);
        CHECK(corr > 0.99, "residual-block gradient correlation = %.5f", corr);

        /* 存取往返也要能保住 blocks */
        CHECK(net_save(&net, "/tmp/goai_net_blk_test.bin"), "residual net_save");
        Net b2; memset(&b2, 0, sizeof(b2));
        CHECK(net_load(&b2, "/tmp/goai_net_blk_test.bin"), "residual net_load");
        CHECK(b2.blocks == BLOCKS && b2.n_params == net.n_params, "blocks and param count preserved");
        int diffs = 0;
        for (int i = 0; i < net.n_params; i++) if (b2.params[i] != net.params[i]) diffs++;
        CHECK(diffs == 0, "residual weights identical after round trip (%d diffs)", diffs);
        net_free(&b2);

        free(grad);
        net_cache_free(&c); net_free(&net);
        free(x); free(pi);
    }

    return 0;
}
