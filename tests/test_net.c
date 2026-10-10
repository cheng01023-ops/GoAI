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

/* 完整目标（策略 + 价值 + 领地）的损失，供有限差分用 */
static float example_loss_ex(Net *net, NetCache *c, const float *x, const float *pi, float z,
                             const float *own, float own_w, float *grad) {
    float pl = 0, vl = 0, ol = 0;
    net_forward(net, c, x, NULL, NULL);
    return net_backward_ex(net, c, pi, z, own, own_w, grad, &pl, &vl, &ol);
}

/* 老格式文件的头部（net.c 里的私有结构，测试里按同样布局写出来） */
typedef struct { uint32_t magic; int32_t size, planes, channels, vhidden, n_params; } HdrOld;
typedef struct { uint32_t magic; int32_t size, planes, channels, vhidden, field, n_params; } HdrNew;

static int write_legacy_net(const char *path, const Net *src, int32_t field) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    if (field < 0) {   /* GOAI：没有 blocks 字段 */
        HdrOld h = { NET_MAGIC, (int32_t)src->size, (int32_t)src->planes, (int32_t)src->channels,
                     (int32_t)src->vhidden, (int32_t)src->n_params };
        if (fwrite(&h, sizeof h, 1, f) != 1) { fclose(f); return 0; }
    } else {           /* GOAJ 老格式：字段就是 blocks */
        HdrNew h = { NET_MAGIC2, (int32_t)src->size, (int32_t)src->planes, (int32_t)src->channels,
                     (int32_t)src->vhidden, field, (int32_t)src->n_params };
        if (fwrite(&h, sizeof h, 1, f) != 1) { fclose(f); return 0; }
    }
    const size_t n = fwrite(src->params, sizeof(float), (size_t)src->n_params, f);
    fclose(f);
    return n == (size_t)src->n_params;
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

    /* ================= 第 ⑤ 步：领地辅助头 ================= */
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Net net; NetCache c;
        net_init_full(&net, size, P, C, H, 0, 1, 0, 41);
        CHECK(net.own_head == 1 && net.vdist == 0, "领地头已启用（值头仍是标量）");
        CHECK(net.len_ow == C && net.len_ob == 1, "领地头参数形状：C 个权重 + 1 个偏置");
        CHECK(net.n_params == net.off_ob + 1, "领地头排在参数区最后");
        /* 生产初始化是很小的权重（sigmoid 不饱和），梯度也小；
           有限差分要看清数学，先把领地头调成普通量级。 */
        Rng rw; rng_seed(&rw, 99);
        for (int i = 0; i < net.len_ow; i++) net.params[net.off_ow + i] = (float)(rng_normal(&rw) * 0.3);
        for (int i = 0; i < net.len_ob; i++) net.params[net.off_ob + i] = (float)(rng_normal(&rw) * 0.2);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;

        Rng rng; rng_seed(&rng, 31337);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        float *own = (float *)calloc((size_t)nn, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = rng_double(&rng) < 0.4 ? 1.0f : 0.0f;
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);
        /* 领地目标混合三种值：己方 / 对方 / 中立(0.5) */
        for (int i = 0; i < nn; i++) own[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? 0.0f : 0.5f;

        const float OWN_W = 0.15f;
        const float z = 1.0f;
        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        float pol[BOARD_MAX_POINTS + 1], v;
        net_zero_grad(&net, grad);
        net_forward(&net, &c, x, pol, &v);
        float l1 = 0, l2 = 0, l3 = 0;
        const float total = net_backward_ex(&net, &c, pi, z, own, OWN_W, grad, &l1, &l2, &l3);
        CHECK(l3 > 0.0f && l3 < 2.0f, "领地 BCE 有限且合理 (%.4f)", (double)l3);
        CHECK(fabsf(total - (l1 + l2 + OWN_W * l3)) < 1e-3f,
              "总损失 = 策略 + 价值 + OWN_WEIGHT*领地 (%.5f)", (double)total);
        {
            double gw = 0;
            for (int i = 0; i < net.len_ow; i++) gw += fabs(grad[net.off_ow + i]);
            CHECK(gw > 0.0, "领地头拿到非零梯度 (%.2e)", gw);
        }
        /* 关闭权重时，领地头和梯度都必须完全不动 */
        net_zero_grad(&net, grad);
        net_forward(&net, &c, x, pol, &v);
        net_backward_ex(&net, &c, pi, z, own, 0.0f, grad, &l1, &l2, &l3);
        {
            double gw = 0;
            for (int i = 0; i < net.len_ow; i++) gw += fabs(grad[net.off_ow + i]);
            CHECK(l3 == 0.0f && gw == 0.0, "OWN_WEIGHT=0 时领地损失和梯度都为 0");
        }
        /* 有限差分（领地头）：头的输入是固定的 h（不再过 ReLU），
           所以这条路径完全光滑，可以要求"最大偏差 ~0"。
           抽全部 C+1 个参数逐个核对。 */
        net_zero_grad(&net, grad);
        example_loss_ex(&net, &c, x, pi, z, own, OWN_W, grad);
        {
            const float h = 1e-2f;
            int checked = 0, bad = 0;
            double worst = 0.0;
            for (int i = net.off_ow; i < net.off_ob + net.len_ob; i++) {
                const float w0 = net.params[i];
                net.params[i] = w0 + h;
                const float lp = example_loss_ex(&net, &c, x, pi, z, own, OWN_W, NULL);
                net.params[i] = w0 - h;
                const float lm = example_loss_ex(&net, &c, x, pi, z, own, OWN_W, NULL);
                net.params[i] = w0;
                const double num = (double)(lp - lm) / (2.0 * h);
                const double an = grad[i];
                checked++;
                if (fabs(num - an) > worst) worst = fabs(num - an);
                if (fabs(num - an) > 2e-3 + 0.05 * fabs(an)) bad++;
            }
            printf("    [领地头梯度] 领地头全部 %d 个参数，最大绝对偏差 %.2e, 超差 %d\n",
                   checked, worst, bad);
            CHECK(bad == 0, "领地头梯度与有限差分一致 (%d/%d 超差)", bad, checked);
            CHECK(worst < 5e-3, "领地头的最大绝对偏差 ~0 (%.2e)", worst);
        }
        /* 干路上的参数（conv 权重）要受 ReLU 折点影响，有限差分会有个别点不准，
           所以这里只查整体相关性（这也验证领地梯度确实回传到了主干）。 */
        {
            const float h = 1e-3f;
            int checked = 0, bad = 0;
            double sum_an = 0, sum_num = 0, sum_a2 = 0, sum_n2 = 0, sum_an2 = 0;
            for (int t = 0; t < 60; t++) {
                const int i = (int)rng_below(&rng, (uint32_t)net.off_pw);   /* 只抽卷积层 */
                const float w0 = net.params[i];
                net.params[i] = w0 + h;
                const float lp = example_loss_ex(&net, &c, x, pi, z, own, OWN_W, NULL);
                net.params[i] = w0 - h;
                const float lm = example_loss_ex(&net, &c, x, pi, z, own, OWN_W, NULL);
                net.params[i] = w0;
                const double num = (double)(lp - lm) / (2.0 * h);
                const double an = grad[i];
                checked++;
                if (fabs(num - an) > 2e-2 + 0.10 * fabs(an)) bad++;
                sum_an += an; sum_num += num; sum_a2 += an * an; sum_n2 += num * num; sum_an2 += an * num;
            }
            const double denom = sqrt((sum_a2 - sum_an * sum_an / checked) * (sum_n2 - sum_num * sum_num / checked));
            const double corr = denom > 0 ? (sum_an2 - sum_an * sum_num / checked) / denom : 0.0;
            CHECK(bad == 0, "主干卷积权重梯度与有限差分一致 (%d/%d 超差)", bad, checked);
            CHECK(corr > 0.99, "主干梯度相关性（含领地项） = %.5f", corr);
        }
        /* 领地梯度确实进了主干：开/关领地损失时 conv2 的梯度必须不同 */
        {
            float *g0 = (float *)calloc((size_t)net.n_params, sizeof(float));
            net_zero_grad(&net, g0);
            net_forward(&net, &c, x, pol, &v);
            net_backward_ex(&net, &c, pi, z, NULL, 0.0f, g0, &l1, &l2, &l3);
            double d = 0;
            for (int i = 0; i < net.len_c2w; i++)
                d += fabs(grad[net.off_c2w + i] - g0[net.off_c2w + i]);
            CHECK(d > 0.0, "领地 BCE 的梯度回传到了主干 conv2 (%.3e)", d);
            free(g0);
        }
        /* 没有领地目标时（own = NULL）等价于关闭 */
        net_zero_grad(&net, grad);
        net_forward(&net, &c, x, pol, &v);
        net_backward_ex(&net, &c, pi, z, NULL, OWN_W, grad, &l1, &l2, &l3);
        {
            double gw = 0;
            for (int i = 0; i < net.len_ow; i++) gw += fabs(grad[net.off_ow + i]);
            CHECK(l3 == 0.0f && gw == 0.0, "own_target=NULL 时不算领地损失");
        }
        free(grad); free(x); free(pi); free(own);
        net_cache_free(&c); net_free(&net);
    }

    /* ================= 第 ⑤ 步：值目标分布化 ================= */
    {
        CHECK(net_value_bucket_index(-1.0f) == 0 && net_value_bucket_index(0.0f) == 16 &&
              net_value_bucket_index(1.0f) == 32, "z=-1/0/+1 落在第 0/16/32 个桶");
        CHECK(net_value_bucket_index(3.0f) == 32 && net_value_bucket_index(-3.0f) == 0,
              "超范围的 z 被夹到两端");
        CHECK(fabsf(net_value_bucket_center(0) + 1.0f) < 1e-6f &&
              fabsf(net_value_bucket_center(32) - 1.0f) < 1e-6f &&
              fabsf(net_value_bucket_center(16)) < 1e-6f, "桶心均匀覆盖 [-1, +1]");
        for (int k = 1; k < NET_VALUE_BUCKETS; k++)
            CHECK(net_value_bucket_center(k) > net_value_bucket_center(k - 1),
                  "桶心单调递增 (k=%d)", k);
    }
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Net net; NetCache c;
        net_init_full(&net, size, P, C, H, 0, 0, 1, 202);
        CHECK(net.vdist == 1 && net.nbuckets == NET_VALUE_BUCKETS, "值分布头：%d 个桶", net.nbuckets);
        CHECK(net.len_vfc2w == H * NET_VALUE_BUCKETS && net.len_vfc2b == NET_VALUE_BUCKETS,
              "值头从 H 变成 H*K");
        Net plain; net_init_ex(&plain, size, P, C, H, 0, 202);
        CHECK(net.n_params - plain.n_params == (H + 1) * (NET_VALUE_BUCKETS - 1),
              "分布化多出的参数 = (H+1)*(K-1)");
        net_free(&plain);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;

        Rng rng; rng_seed(&rng, 4242);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = rng_double(&rng) < 0.4 ? 1.0f : 0.0f;
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);

        float pol[BOARD_MAX_POINTS + 1], v;
        net_forward(&net, &c, x, pol, &v);
        {
            double ps = 0, e = 0;
            for (int k = 0; k < NET_VALUE_BUCKETS; k++) {
                ps += c.vprobs[k];
                e += (double)c.vprobs[k] * (double)net_value_bucket_center(k);
            }
            CHECK(fabs(ps - 1.0) < 1e-5, "值分布 softmax 归一 (%.6f)", ps);
            CHECK(fabs(e - (double)v) < 1e-5, "搜索值 = 分布期望 (%.6f vs %.6f)", e, (double)v);
            CHECK(v > -1.0f && v < 1.0f, "分布期望落在 (-1,1)");
        }
        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        float l1 = 0, l2 = 0, l3 = 0;
        net_zero_grad(&net, grad);
        net_backward_ex(&net, &c, pi, 1.0f, NULL, 0.0f, grad, &l1, &l2, &l3);
        CHECK(fabs((double)l2 + log((double)c.vprobs[32])) < 1e-4,
              "值损失 = -log p(z 所在桶) (%.6f)", (double)l2);
        CHECK(l3 == 0.0f, "没有领地头时领地损失为 0");

        /* 有限差分（值分布头）：logits = W·ha + b 是线性的、损失是 softmax 交叉熵，
           全程光滑，所以要求"最大偏差 ~0"；抽头里的 60 个参数逐个核对。 */
        net_zero_grad(&net, grad);
        example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, grad);
        {
            const float h = 1e-2f;
            int checked = 0, bad = 0;
            double worst = 0.0;
            const int lo = net.off_vfc2w, hi = net.off_vfc2b + net.len_vfc2b;
            for (int t = 0; t < 60; t++) {
                const int i = lo + (int)rng_below(&rng, (uint32_t)(hi - lo));
                const float w0 = net.params[i];
                net.params[i] = w0 + h;
                const float lp = example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, NULL);
                net.params[i] = w0 - h;
                const float lm = example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, NULL);
                net.params[i] = w0;
                const double num = (double)(lp - lm) / (2.0 * h);
                const double an = grad[i];
                checked++;
                if (fabs(num - an) > worst) worst = fabs(num - an);
                if (fabs(num - an) > 2e-3 + 0.05 * fabs(an)) bad++;
            }
            printf("    [值分布头梯度] 最大绝对偏差 %.2e, 超差 %d/%d\n", worst, bad, checked);
            CHECK(bad == 0, "值分布头梯度与有限差分一致 (%d/%d 超差)", bad, checked);
            CHECK(worst < 5e-3, "值分布头的最大绝对偏差 ~0 (%.2e)", worst);
        }
        /* 干路（conv）受 ReLU 折点影响：只查整体相关性 */
        {
            const float h = 1e-3f;
            int checked = 0, bad = 0;
            double sum_an = 0, sum_num = 0, sum_a2 = 0, sum_n2 = 0, sum_an2 = 0;
            for (int t = 0; t < 60; t++) {
                const int i = (int)rng_below(&rng, (uint32_t)net.off_pw);
                const float w0 = net.params[i];
                net.params[i] = w0 + h;
                const float lp = example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, NULL);
                net.params[i] = w0 - h;
                const float lm = example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, NULL);
                net.params[i] = w0;
                const double num = (double)(lp - lm) / (2.0 * h);
                const double an = grad[i];
                checked++;
                if (fabs(num - an) > 2e-2 + 0.10 * fabs(an)) bad++;
                sum_an += an; sum_num += num; sum_a2 += an * an; sum_n2 += num * num; sum_an2 += an * num;
            }
            const double denom = sqrt((sum_a2 - sum_an * sum_an / checked) * (sum_n2 - sum_num * sum_num / checked));
            const double corr = denom > 0 ? (sum_an2 - sum_an * sum_num / checked) / denom : 0.0;
            CHECK(bad == 0, "值分布主干卷积梯度与有限差分一致 (%d/%d 超差)", bad, checked);
            CHECK(corr > 0.99, "值分布主干梯度相关性 = %.5f", corr);
        }
        free(grad); free(x); free(pi);
        net_cache_free(&c); net_free(&net);
    }
    {
        /* 分布头训练：期望值应朝 +1 走、交叉熵下降 */
        const int size = 9, P = NET_FEATURE_PLANES, C = 8, H = 16;
        Net net; NetCache c;
        net_init_full(&net, size, P, C, H, 0, 0, 1, 17);
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
        const float v0 = v;
        const float loss0 = -logf(c.vprobs[32] > 1e-12f ? c.vprobs[32] : 1e-12f);
        for (int step = 1; step <= 300; step++) {
            net_zero_grad(&net, grad);
            example_loss_ex(&net, &c, x, pi, 1.0f, NULL, 0.0f, grad);
            net_adam(&net, grad, m, vv, step, 0.02f, 0.0f, 5.0f);
        }
        net_forward(&net, &c, x, pol, &v);
        const float loss1 = -logf(c.vprobs[32] > 1e-12f ? c.vprobs[32] : 1e-12f);
        CHECK(v > v0 + 0.2f, "值分布期望朝 +1 移动 (%.3f -> %.3f)", (double)v0, (double)v);
        CHECK(loss1 < loss0 * 0.5f, "值分布交叉熵下降 (%.3f -> %.3f)", (double)loss0, (double)loss1);
        free(grad); free(m); free(vv); free(x); free(pi);
        net_cache_free(&c); net_free(&net);
    }

    /* ================= 残差块 + 两个新头一起用（回归：头部 1x1 的输入）================= */
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8, BLOCKS = 2;
        Net net; NetCache c;
        net_init_full(&net, size, P, C, H, BLOCKS, 1, 1, 77);
        CHECK(net.blocks == BLOCKS && net.own_head == 1 && net.vdist == 1,
              "残差块 + 领地头 + 值分布可以同时存在");
        /* 块零初始化会让块的梯度退化成 0，这里随机化以便校验数学 */
        Rng rw; rng_seed(&rw, 4242);
        for (int i = 0; i < net.len_bw; i++) net.params[net.off_bw + i] = (float)(rng_normal(&rw) * 0.08);
        for (int i = 0; i < net.len_ow; i++) net.params[net.off_ow + i] = (float)(rng_normal(&rw) * 0.3);
        net_cache_init(&net, &c);
        const int nn = size * size, NP = nn + 1;

        Rng rng; rng_seed(&rng, 555);
        float *x = (float *)calloc((size_t)P * nn, sizeof(float));
        float *pi = (float *)calloc((size_t)NP, sizeof(float));
        float *own = (float *)calloc((size_t)nn, sizeof(float));
        for (int i = 0; i < P * nn; i++) x[i] = rng_double(&rng) < 0.4 ? 1.0f : 0.0f;
        double s = 0;
        for (int i = 0; i < NP; i++) { pi[i] = (float)(rng_double(&rng) + 0.05); s += pi[i]; }
        for (int i = 0; i < NP; i++) pi[i] = (float)(pi[i] / s);
        for (int i = 0; i < nn; i++) own[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1) ? 0.0f : 0.5f;

        float *grad = (float *)calloc((size_t)net.n_params, sizeof(float));
        net_zero_grad(&net, grad);
        example_loss_ex(&net, &c, x, pi, 1.0f, own, 0.15f, grad);
        /* 三个头部的 1x1 卷积权重：前向输入是"最后一块的输出"，
           反向也必须用同一个缓存（否则有残差块时这 17 个参数的梯度全错） */
        struct { const char *name; int lo, hi; } segs[3];
        segs[0].name = "策略头 1x1"; segs[0].lo = net.off_pw; segs[0].hi = net.off_pw + net.len_pw;
        segs[1].name = "价值头 1x1"; segs[1].lo = net.off_vw; segs[1].hi = net.off_vw + net.len_vw;
        segs[2].name = "领地头 1x1"; segs[2].lo = net.off_ow; segs[2].hi = net.off_ob + net.len_ob;
        const float h = 1e-3f;
        for (int k = 0; k < 3; k++) {
            double worst = 0.0;
            int bad = 0, n = 0;
            for (int i = segs[k].lo; i < segs[k].hi; i++) {
                const float w0 = net.params[i];
                net.params[i] = w0 + h;
                const float lp = example_loss_ex(&net, &c, x, pi, 1.0f, own, 0.15f, NULL);
                net.params[i] = w0 - h;
                const float lm = example_loss_ex(&net, &c, x, pi, 1.0f, own, 0.15f, NULL);
                net.params[i] = w0;
                const double num = (double)(lp - lm) / (2.0 * h);
                const double an = grad[i];
                n++;
                if (fabs(num - an) > worst) worst = fabs(num - an);
                if (fabs(num - an) > 2e-3 + 0.05 * fabs(an)) bad++;
            }
            printf("    [残差块头部] %s：%d 个参数，最大偏差 %.2e，超差 %d\n",
                   segs[k].name, n, worst, bad);
            CHECK(bad == 0, "%s 的梯度在有残差块时也正确 (%d/%d 超差)", segs[k].name, bad, n);
            CHECK(worst < 5e-3, "%s 最大偏差 ~0 (%.2e)", segs[k].name, worst);
        }
        free(grad); free(x); free(pi); free(own);
        net_cache_free(&c); net_free(&net);
    }

    /* ================= 格式兼容：flags / 老文件 ================= */
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        /* 新格式往返：残差块 + 领地头 + 值分布 */
        Net a; net_init_full(&a, size, P, C, H, 2, 1, 1, 9);
        Rng rw; rng_seed(&rw, 5);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rw) * 0.01);
        CHECK(net_save(&a, "/tmp/goai_net_flags.bin"), "新格式(flags) net_save");
        Net b; memset(&b, 0, sizeof(b));
        CHECK(net_load(&b, "/tmp/goai_net_flags.bin"), "新格式(flags) net_load");
        CHECK(b.blocks == 2 && b.own_head == 1 && b.vdist == 1, "blocks/领地头/值分布都保住了");
        CHECK(b.n_params == a.n_params, "参数个数一致 (%d)", b.n_params);
        {
            int diff = 0;
            for (int i = 0; i < a.n_params; i++) if (a.params[i] != b.params[i]) diff++;
            CHECK(diff == 0, "新格式往返权重逐个相同 (%d 个不同)", diff);
        }
        net_free(&b); net_free(&a);
    }
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Rng rw; rng_seed(&rw, 11);
        /* GOAI 老文件：2 层卷积、没有 blocks 字段 */
        Net a; net_init(&a, size, P, C, H, 5);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rw) * 0.02);
        CHECK(write_legacy_net("/tmp/goai_net_goai.bin", &a, -1), "写出 GOAI 老文件");
        Net b; memset(&b, 0, sizeof(b));
        CHECK(net_load(&b, "/tmp/goai_net_goai.bin"), "GOAI 老文件仍能加载");
        CHECK(b.blocks == 0 && b.own_head == 0 && b.vdist == 0, "GOAI 解释成无额外头的老网络");
        CHECK(b.n_params == a.n_params, "GOAI 参数个数一致");
        {
            int diff = 0;
            for (int i = 0; i < a.n_params; i++) if (a.params[i] != b.params[i]) diff++;
            CHECK(diff == 0, "GOAI 权重加载正确 (%d 个不同)", diff);
        }
        net_free(&b); net_free(&a);
    }
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Rng rw; rng_seed(&rw, 13);
        /* GOAJ 老文件（blocks = 2，偶数：低 8 位为 0，天然兼容） */
        Net a; net_init_ex(&a, size, P, C, H, 2, 5);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rw) * 0.02);
        CHECK(write_legacy_net("/tmp/goai_net_goaj2.bin", &a, 2), "写出 GOAJ(blocks=2) 老文件");
        Net b; memset(&b, 0, sizeof(b));
        CHECK(net_load(&b, "/tmp/goai_net_goaj2.bin"), "GOAJ(blocks=2) 老文件仍能加载");
        CHECK(b.blocks == 2 && b.own_head == 0 && b.vdist == 0, "GOAJ blocks=2 解释正确");
        CHECK(b.n_params == a.n_params, "GOAJ(blocks=2) 参数个数一致");
        net_free(&b); net_free(&a);
    }
    {
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Rng rw; rng_seed(&rw, 17);
        /* GOAJ 老文件（blocks = 3，奇数：会撞上 flags 的 bit0，靠 n_params 交叉核对救回来） */
        Net a; net_init_ex(&a, size, P, C, H, 3, 5);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rw) * 0.02);
        CHECK(write_legacy_net("/tmp/goai_net_goaj3.bin", &a, 3), "写出 GOAJ(blocks=3) 老文件");
        Net b; memset(&b, 0, sizeof(b));
        CHECK(net_load(&b, "/tmp/goai_net_goaj3.bin"), "GOAJ(blocks=3) 老文件仍能加载");
        CHECK(b.blocks == 3 && b.own_head == 0 && b.vdist == 0,
              "奇数 blocks 的老文件被正确识别（n_params 交叉核对）");
        {
            int diff = 0;
            for (int i = 0; i < a.n_params; i++) if (a.params[i] != b.params[i]) diff++;
            CHECK(diff == 0, "GOAJ(blocks=3) 权重加载正确 (%d 个不同)", diff);
        }
        net_free(&b); net_free(&a);
    }
    {
        /* 内置权重（net_load_mem，GOAI 格式）仍然能用 */
        const int size = 9, P = NET_FEATURE_PLANES, C = 8, H = 16;
        Net a; net_init(&a, size, P, C, H, 5);
        Rng rw; rng_seed(&rw, 23);
        for (int i = 0; i < a.n_params; i++) a.params[i] += (float)(rng_normal(&rw) * 0.02);
        CHECK(write_legacy_net("/tmp/goai_net_mem.bin", &a, -1), "写出内存加载用的老文件");
        FILE *f = fopen("/tmp/goai_net_mem.bin", "rb");
        void *buf = malloc((size_t)(sizeof(HdrOld) + (size_t)a.n_params * sizeof(float)));
        const size_t rd = fread(buf, 1, sizeof(HdrOld) + (size_t)a.n_params * sizeof(float), f);
        fclose(f);
        Net b; memset(&b, 0, sizeof(b));
        CHECK(rd > 0 && net_load_mem(&b, buf, rd), "net_load_mem 仍能读老格式内置权重");
        CHECK(b.own_head == 0 && b.vdist == 0 && b.n_params == a.n_params, "内置权重解释正确");
        free(buf); net_free(&b); net_free(&a);
    }
    {
        /* net_copy_shared：标量值头 -> 分布值头时形状不同，不能硬拷（会越界） */
        const int size = 5, P = NET_FEATURE_PLANES, C = 4, H = 8;
        Net src; net_init_ex(&src, size, P, C, H, 0, 21);
        Net dst; net_init_full(&dst, size, P, C, H, 0, 1, 1, 22);
        Rng rw; rng_seed(&rw, 29);
        for (int i = 0; i < src.n_params; i++) src.params[i] += (float)(rng_normal(&rw) * 0.05);
        net_copy_shared(&dst, &src);
        CHECK(src.len_vfc2w == H && dst.len_vfc2w == H * NET_VALUE_BUCKETS,
              "两边的值头形状确实不同 (H=%d vs H*K=%d)", src.len_vfc2w, dst.len_vfc2w);
        {
            int same = 1;
            for (int i = 0; i < dst.len_c1w; i++)
                if (dst.params[dst.off_c1w + i] != src.params[src.off_c1w + i]) same = 0;
            CHECK(same, "形状相同的卷积权重被完整搬过去");
            int same_top = 1;
            for (int i = 0; i < dst.len_pfcw; i++)
                if (dst.params[dst.off_pfcw + i] != src.params[src.off_pfcw + i]) same_top = 0;
            CHECK(same_top, "策略全连接权重被完整搬过去");
        }
        /* 拷贝之后网络结构不变，前向仍然正常 */
        NetCache c; net_cache_init(&dst, &c);
        float x[4 * 25], pol[26], v;
        Board bd; board_init(&bd, 5);
        net_features(&bd, x);
        net_forward(&dst, &c, x, pol, &v);
        CHECK(v > -1.0f && v < 1.0f, "迁移后的分布网络前向正常 (v=%.3f)", (double)v);
        net_cache_free(&c); net_free(&src); net_free(&dst);
    }

    return 0;
}
