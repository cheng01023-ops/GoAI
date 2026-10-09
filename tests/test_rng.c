#include "rand.h"
#include "test_util.h"
#include <string.h>

/* 随机数是搜索树的索引来源：一旦越界就会写坏内存，必须严格验证 */
int test_rng_run(void) {
    Rng r;
    rng_seed(&r, 0xC0FFEEULL);

    /* 1) 范围：对 1..1024 各种 n 反复抽样，结果必须落在 [0,n) */
    int out_of_range = 0;
    for (uint32_t n = 1; n <= 1024; n++) {
        for (int k = 0; k < 200; k++) {
            const uint32_t v = rng_below(&r, n);
            if (v >= n) out_of_range++;
        }
    }
    CHECK(out_of_range == 0, "rng_below 越界 %d 次（1..1024 各 200 次）", out_of_range);

    /* 2) n = 0 安全返回 0，不消耗随机流 */
    {
        Rng a, b;
        rng_seed(&a, 1); rng_seed(&b, 1);
        CHECK(rng_below(&a, 0) == 0, "n=0 返回 0");
        CHECK(rng_next(&a) == rng_next(&b), "n=0 不消耗随机流");
    }

    /* 3) 均匀性：0..9 十万次，偏差 < 3% */
    {
        int bucket[10];
        memset(bucket, 0, sizeof(bucket));
        for (int i = 0; i < 100000; i++) bucket[rng_below(&r, 10)]++;
        int worst = 0;
        for (int i = 0; i < 10; i++) {
            const int d = bucket[i] > 10000 ? bucket[i] - 10000 : 10000 - bucket[i];
            if (d > worst) worst = d;
        }
        CHECK(worst < 300, "均匀性偏差 %d/10000 (<3%%)", worst);
    }

    /* 4) 与官方 128 位参考实现逐位对比（仅拒绝采样时可能不同）
     *    注意：__uint128_t 是 GCC/Clang 扩展，MSVC 没有，故做条件编译 */
#if defined(__SIZEOF_INT128__)
    {
        Rng a, b;
        rng_seed(&a, 99); rng_seed(&b, 99);
        int mismatch = 0;
        for (int i = 0; i < 100000; i++) {
            const uint32_t n = (uint32_t)(1 + (i % 361));
            const uint32_t x = rng_below(&a, n);
            const uint32_t y = (uint32_t)(((__uint128_t)rng_next(&b) * (__uint128_t)n) >> 64);
            if (x != y) mismatch++;
        }
        CHECK(mismatch < 50, "与 128 位参考实现差异 %d/100000（应仅来自拒绝采样）", mismatch);
    }
#else
    CHECK(1, "跳过 128 位参考实现对比（当前编译器不支持 __uint128_t）");
#endif

    /* 5) 棋盘规模的取值上限（9/13/19 路点数与小于它们的合法数） */
    {
        Rng s;
        rng_seed(&s, 5);
        int bad = 0;
        for (int i = 0; i < 20000; i++) {
            if (rng_below(&s, 81) > 80) bad++;
            if (rng_below(&s, 361) > 360) bad++;
            if (rng_below(&s, 3) > 2) bad++;
        }
        CHECK(bad == 0, "棋盘规模抽样越界 %d 次", bad);
    }
    return 0;
}
