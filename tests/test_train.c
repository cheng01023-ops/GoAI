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

    /* 7) 第 ⑤ 步：领地标签的登记 / 取用（按棋局号索引，一盘棋只存一份） */
    {
        TrainConfig c = base_cfg();
        c.buffer_cap = 64;
        c.channels = 4;
        c.vhidden = 8;
        c.own_weight = 0.15f;               /* > 0 => 网络带领地头 */
        c.vdist = 1;
        Trainer t;
        CHECK(trainer_init(&t, &c, 7) == 0, "trainer_init（领地头 + 值分布）");
        CHECK(t.net.own_head == 1 && t.net.vdist == 1, "网络确实建了领地头和值分布头");
        CHECK(t.own_tab != NULL && t.own_slots > 0, "领地标签表已分配（%d 槽）", t.own_slots);
        ExampleBatch b;
        CHECK(batch_init(&b, 4, t.xlen, t.pilen) == 0, "batch_init");
        b.n = 2;
        memset(b.X, 0, (size_t)2 * (size_t)t.xlen);
        memset(b.PI, 0, (size_t)2 * (size_t)t.pilen * sizeof(float));
        b.Z[0] = 1.0f; b.Z[1] = -1.0f;
        int8_t own[81];
        for (int i = 0; i < 81; i++) own[i] = (i % 3 == 0) ? 1 : (i % 3 == 1 ? -1 : 0);
        trainer_push_game(&t, &b, own);
        CHECK(t.count == 2, "两个样本都进了回放缓冲");
        const int gid = t.bown[0];
        CHECK(gid > 0 && t.bown[1] == gid, "同一盘棋的样本共享棋局号");
        {
            const int8_t *lab = trainer_own_labels(&t, gid);
            CHECK(lab != NULL && memcmp(lab, own, 81) == 0, "领地标签按棋局号取回且完全一致");
        }
        CHECK(t.bwhite[0] == 0 && t.bwhite[1] == 1, "样本 0 轮到黑、样本 1 轮到白");
        CHECK(trainer_own_labels(&t, gid + 12345) == NULL, "不存在的棋局号返回 NULL");
        batch_free(&b);
        trainer_free(&t);
    }

    /* 8) 关闭额外头时不分配标签表（默认行为完全不变） */
    {
        TrainConfig c = base_cfg();
        c.buffer_cap = 64;
        c.channels = 4;
        c.vhidden = 8;
        Trainer t;
        CHECK(trainer_init(&t, &c, 8) == 0, "trainer_init（默认配置）");
        CHECK(t.net.own_head == 0 && t.net.vdist == 0 && t.own_tab == NULL,
              "默认不建额外头、不分配标签表");
        trainer_free(&t);
    }
    return 0;
}
