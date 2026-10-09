/* apps.h - command line front ends: GTP, human play, self-play export, evaluation */
#ifndef GOAI_APPS_H
#define GOAI_APPS_H

#include "train.h"

int gtp_main(const char *weights, int sims, int size, double komi, uint64_t seed);
int human_play_main(const char *weights, int sims, int size, double komi, int human_color);
int selfplay_main(const char *weights, int games, int sims, int size, double komi,
                  const char *out_dir, uint64_t seed);
int diag_main(const char *weights, int size);
void eval_set_open_seed(uint64_t s);
int eval_main(const char *a_spec, const char *b_spec, int games, int sims, int size, int open_plies,
              double komi, uint64_t seed);

#endif /* GOAI_APPS_H */
