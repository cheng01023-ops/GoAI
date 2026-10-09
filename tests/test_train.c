/* test_train.c - 学习率衰减计划的单元测试 */
#include <math.h>
#include <string.h>

#include "test_util.h"
#include "train.h"

static TrainConfig base_cfg(void) {
    TrainConfig c;
    train_config_default(&c, 9);
    c.lr = 0.02f;
    c.lr_min = 1e-4f;
    c.lr_decay_every = 0;
    c.lr_decay_factor = 0.7f;
    return c;
}

int test_train_run(void) {
    /* 1) 关闭衰减时学习率恒定 */
    {
        TrainConfig c = base_cfg();
        CHECK(fabsf(train_lr_at_iter(&c, 0) - 0.02f) < 1e-9f, "无衰减 iter0 应为 0.02");
        CHECK(fabsf(train_lr_at_iter(&c, 999) - 0.02f) < 1e-9f, "无衰减 iter999 应恒定");
    }

    /* 2) 每 200 轮 ×0.7 */
    {
        TrainConfig c = base_cfg();
        c.lr_decay_every = 200;
        CHECK(fabsf(train_lr_at_iter(&c, 0) - 0.02f) < 1e-9f, "第 0 轮不衰减");
        CHECK(fabsf(train_lr_at_iter(&c, 199) - 0.02f) < 1e-9f, "第 199 轮仍不衰减");
        CHECK(fabsf(train_lr_at_iter(&c, 200) - 0.02f * 0.7f) < 1e-9f, "第 200 轮应为 0.7 倍");
        CHECK(fabsf(train_lr_at_iter(&c, 400) - 0.02f * 0.49f) < 1e-9f, "第 400 轮应为 0.49 倍");
        CHECK(fabsf(train_lr_at_iter(&c, 600) - 0.02f * 0.343f) < 1e-7f, "第 600 轮应为 0.343 倍");
    }

    /* 3) 下限夹紧 */
    {
        TrainConfig c = base_cfg();
        c.lr_decay_every = 50;
        c.lr_min = 0.005f;
        CHECK(fabsf(train_lr_at_iter(&c, 5000) - 0.005f) < 1e-9f, "应夹在下限 0.005");
        CHECK(train_lr_at_iter(&c, 5000) >= c.lr_min, "任何轮次都不低于下限");
        CHECK(train_lr_at_iter(&c, 600000) >= c.lr_min, "极大轮次也不低于下限");
    }

    /* 4) 非法参数不衰减 */
    {
        TrainConfig c = base_cfg();
        c.lr_decay_every = 100;
        c.lr_decay_factor = 1.0f;
        CHECK(fabsf(train_lr_at_iter(&c, 500) - 0.02f) < 1e-9f, "factor=1 不衰减");
        c.lr_decay_factor = 0.0f;
        CHECK(fabsf(train_lr_at_iter(&c, 500) - 0.02f) < 1e-9f, "factor=0 视为关闭");
        c.lr_decay_factor = 1.5f;
        CHECK(fabsf(train_lr_at_iter(&c, 500) - 0.02f) < 1e-9f, "factor>1 视为关闭");
        c.lr_decay_factor = 0.7f;
        c.lr_decay_every = -5;
        CHECK(fabsf(train_lr_at_iter(&c, 500) - 0.02f) < 1e-9f, "负数间隔视为关闭");
    }

    /* 5) 单调不增 */
    {
        TrainConfig c = base_cfg();
        c.lr_decay_every = 100;
        float prev = train_lr_at_iter(&c, 0);
        int ok = 1;
        for (int it = 1; it <= 2000; it++) {
            const float lr = train_lr_at_iter(&c, it);
            if (lr > prev + 1e-9f) ok = 0;
            prev = lr;
        }
        CHECK(ok, "学习率必须随轮次单调不增");
    }

    /* 6) 胜率置信区间 */
    {
        const double ci20 = train_winrate_ci(0.5, 20);
        const double ci80 = train_winrate_ci(0.5, 80);
        CHECK(ci20 > 0.20 && ci20 < 0.23, "20 局 50%% 胜率 CI 约 +-0.22 (实得 %.3f)", ci20);
        CHECK(ci80 > 0.10 && ci80 < 0.12, "80 局 CI 约 +-0.11 (实得 %.3f)", ci80);
        CHECK(ci80 < ci20, "局数越多置信区间越窄");
        CHECK(train_winrate_ci(0.0, 20) == 0.0, "全负时 CI 为 0");
        CHECK(train_winrate_ci(1.0, 20) == 0.0, "全胜时 CI 为 0");
        CHECK(train_winrate_ci(0.5, 0) == 0.0, "0 局时 CI 为 0");
        CHECK(train_winrate_ci(2.0, 10) == train_winrate_ci(1.0, 10), "胜率夹到 [0,1]");
    }
    return 0;
}
