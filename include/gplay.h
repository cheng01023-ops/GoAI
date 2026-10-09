/* gplay.h - GPU 服务驱动的批量自对弈（C 搜索 + GPU 批量推理） */
#ifndef GOAI_GPLAY_H
#define GOAI_GPLAY_H

#include "remote.h"
#include "train.h"

/* 每个线程跑 slots 盘棋；所有线程共用一个 GPU 服务连接池 */
int gplay_run(const char *addr, int threads, int slots_per_thread, int sims,
              int size, double komi, int max_moves, int pass_min_move,
              int temp_moves, uint64_t seed, const char *out_dir, int save_sgf);

#endif /* GOAI_GPLAY_H */
