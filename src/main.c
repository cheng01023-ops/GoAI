/* main.c - GoAI command line entry point */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "apps.h"
#include "gplay.h"
#include "train.h"

#define GOAI_VERSION "1.0"

static void usage(void) {
    printf(
        "GoAI %s - a Go engine written in C (MCTS + neural network, trained by self-play)\n"
        "\n"
        "usage: GoAI <command> [options]\n"
        "\n"
        "  train      train the network by self-play reinforcement learning\n"
        "               --size N     board size (default 9)\n"
        "               --iters N    training iterations (default 6)\n"
        "               --games N    self-play games per iteration (default 30)\n"
        "               --sims N     MCTS simulations per move (default 120)\n"
        "               --steps N    gradient steps per iteration (default 300)\n"
        "               --batch N    mini-batch size (default 64)\n"
        "               --lr X       learning rate (default 0.02)\n"
        "               --lr-decay-every N   halve-ish the LR every N rounds (0 = off)\n"
        "               --lr-decay-factor F  multiply LR by F each decay (default 0.7)\n"
        "               --lr-min X           never decay below X (default 1e-4)\n"
        "               --anchor F.bin       fixed reference net for a low-noise progress curve\n"
        "               --anchor-every N     evaluate against it every N rounds (default 20)\n"
        "               --anchor-games N     games per anchor evaluation (default 20)\n"
        "               --open-plies N       paired openings for matchplay (0=off, default 4)\n"
        "               --reuse 0/1          reuse search tree between moves (default 0)\n"
        "               --rollback 0/1       fall back to best.bin when training degrades (default 1)\n"
        "               --rollback-patience N  consecutive bad gates before rollback (default 3)\n"
        "               --channels N conv channels (default 16 for 9x9)\n"
        "               --buffer N   replay buffer size (default 30000)\n"
        "               --evalgames N evaluation games per iteration (default 20)\n"
        "               --evalsims N  simulations for evaluation games (default 40)\n"
        "               --out DIR    output directory (default runs)\n"
        "               --forever    keep training until stopped (Ctrl-C / 停止脚本)\n"
        "               --resume F   resume from a weights file\n"
        "               --gate 0/1   promote best.bin only when stronger (default 1)\n"
        "               --gategames N --eval-every N   promotion match settings\n"
        "               --plot 0/1   refresh the curve PNG every round\n"        "               --threads N  self-play threads (default: one per CPU, max 8)\n"
        "               --seed N     random seed\n"
        "  bench      measure network / MCTS / self-play speed\n"
        "  gtp        speak GTP on stdin/stdout (usable from Sabaki, GoGui, ...)\n"
        "               --weights FILE   network weights (omit for random-rollout MCTS)\n"
        "               --sims N         simulations per move (default 400)\n"
        "  play       play against the engine in the terminal\n"
        "               --weights FILE   network weights\n"
        "               --sims N         simulations per move (default 400)\n"
        "               --white          you play white\n"
        "  selfplay   generate self-play games and write them as SGF\n"
        "               --weights FILE, --games N, --sims N, --out DIR\n"
        "  eval       play a match: --a SPEC --b SPEC [--games N] [--sims N]\n"
        "               SPEC is a weights file or 'random'\n"
        "  gtrain     C search + GPU batched inference server (recommended for RTX)\n"
        "               --remote HOST:PORT  --threads N  --slots N  --sims N  --out DIR\n"
        "  diag       inspect a trained network on test positions\n"
        "  version    print version\n"
        "\n"
        "examples:\n"
        "  GoAI train --size 9 --iters 8 --games 40 --sims 120\n"
        "  GoAI eval --a runs/latest.bin --b random --games 40 --sims 100\n"
        "  GoAI play --weights runs/latest.bin --sims 400\n",
        GOAI_VERSION);
}

static const char *arg_str(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 2; i + 1 < argc; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}

static int arg_int(int argc, char **argv, const char *flag, int def) {
    const char *v = arg_str(argc, argv, flag, NULL);
    return v ? atoi(v) : def;
}

static double arg_dbl(int argc, char **argv, const char *flag, double def) {
    const char *v = arg_str(argc, argv, flag, NULL);
    return v ? atof(v) : def;
}

static int has_flag(int argc, char **argv, const char *flag) {
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];

    if (!strcmp(cmd, "version")) {
        printf("GoAI %s (C11, MCTS + policy/value network)\n", GOAI_VERSION);
        return 0;
    }

    if (!strcmp(cmd, "train")) {
        TrainConfig cfg;
        train_config_default(&cfg, arg_int(argc, argv, "--size", 9));
        cfg.iterations     = arg_int(argc, argv, "--iters", cfg.iterations);
        cfg.games_per_iter = arg_int(argc, argv, "--games", cfg.games_per_iter);
        cfg.sims           = arg_int(argc, argv, "--sims", cfg.sims);
        cfg.train_steps    = arg_int(argc, argv, "--steps", cfg.train_steps);
        cfg.batch_size     = arg_int(argc, argv, "--batch", cfg.batch_size);
        cfg.lr             = (float)arg_dbl(argc, argv, "--lr", cfg.lr);
        cfg.lr_decay_every = arg_int(argc, argv, "--lr-decay-every", cfg.lr_decay_every);
        cfg.lr_decay_factor= (float)arg_dbl(argc, argv, "--lr-decay-factor", cfg.lr_decay_factor);
        cfg.lr_min         = (float)arg_dbl(argc, argv, "--lr-min", cfg.lr_min);
        cfg.anchor_path    = arg_str(argc, argv, "--anchor", cfg.anchor_path);
        cfg.anchor_every   = arg_int(argc, argv, "--anchor-every", cfg.anchor_every);
        cfg.anchor_games   = arg_int(argc, argv, "--anchor-games", cfg.anchor_games);
        cfg.open_plies     = arg_int(argc, argv, "--open-plies", cfg.open_plies);
        { const char *os = arg_str(argc, argv, "--open-seed", NULL);
          if (os) cfg.open_seed = strtoull(os, NULL, 10); }
        cfg.reuse          = arg_int(argc, argv, "--reuse", cfg.reuse);
        cfg.rollback       = arg_int(argc, argv, "--rollback", cfg.rollback);
        cfg.rollback_patience = arg_int(argc, argv, "--rollback-patience", cfg.rollback_patience);
        cfg.channels       = arg_int(argc, argv, "--channels", cfg.channels);
        cfg.blocks         = arg_int(argc, argv, "--blocks", cfg.blocks);
        cfg.buffer_cap     = arg_int(argc, argv, "--buffer", cfg.buffer_cap);
        cfg.eval_games     = arg_int(argc, argv, "--evalgames", cfg.eval_games);
        cfg.eval_sims      = arg_int(argc, argv, "--evalsims", cfg.eval_sims);
        cfg.save_sgf       = arg_int(argc, argv, "--sgf", cfg.save_sgf);
        cfg.augment        = arg_int(argc, argv, "--augment", cfg.augment);
        cfg.threads        = arg_int(argc, argv, "--threads", cfg.threads);
        if (has_flag(argc, argv, "--forever")) cfg.forever = 1;
        {
            const char *rp = arg_str(argc, argv, "--resume", NULL);
            if (rp) { cfg.resume = 1; cfg.resume_path = rp; }
        }
        cfg.status_file    = arg_str(argc, argv, "--status", cfg.status_file);
        cfg.stop_file      = arg_str(argc, argv, "--stopfile", cfg.stop_file);
        cfg.gate           = arg_int(argc, argv, "--gate", cfg.gate);
        cfg.gate_games     = arg_int(argc, argv, "--gategames", cfg.gate_games);
        cfg.eval_every     = arg_int(argc, argv, "--eval-every", cfg.eval_every);
        cfg.plot           = arg_int(argc, argv, "--plot", cfg.plot);
        cfg.out_dir        = arg_str(argc, argv, "--out", cfg.out_dir);
        const char *sd     = arg_str(argc, argv, "--seed", NULL);
        if (sd) cfg.seed = strtoull(sd, NULL, 10);
        return train_run(&cfg);
    }

    if (!strcmp(cmd, "gtrain")) {
        const char *addr = arg_str(argc, argv, "--remote", "127.0.0.1:8899");
        const int threads = arg_int(argc, argv, "--threads", 4);
        const int slots = arg_int(argc, argv, "--slots", 32);
        const int sims = arg_int(argc, argv, "--sims", 64);
        const int size = arg_int(argc, argv, "--size", 9);
        const double komi = arg_dbl(argc, argv, "--komi", 7.0);
        const int maxmoves = arg_int(argc, argv, "--max-moves", 3 * size * size);
        const int passmin = arg_int(argc, argv, "--pass-min-move", (size * size * 2) / 3);
        const int tempmoves = arg_int(argc, argv, "--temp-moves", 12);
        const char *out = arg_str(argc, argv, "--out", "runs_gpu");
        const int sgf = arg_int(argc, argv, "--sgf", 1);
        const char *sd = arg_str(argc, argv, "--seed", NULL);
        return gplay_run(addr, threads, slots, sims, size, komi, maxmoves, passmin,
                         tempmoves, sd ? strtoull(sd, NULL, 10) : 20241001ULL, out, sgf);
    }

    if (!strcmp(cmd, "diag")) {
        const char *w = arg_str(argc, argv, "--weights", "runs/latest.bin");
        return diag_main(w, arg_int(argc, argv, "--size", 9));
    }

    if (!strcmp(cmd, "bench")) {
        TrainConfig cfg;
        train_config_default(&cfg, arg_int(argc, argv, "--size", 9));
        cfg.sims = arg_int(argc, argv, "--sims", cfg.sims);
        cfg.channels = arg_int(argc, argv, "--channels", cfg.channels);
        cfg.blocks   = arg_int(argc, argv, "--blocks", cfg.blocks);
        return train_bench(&cfg);
    }

    if (!strcmp(cmd, "gtp")) {
        const char *w = arg_str(argc, argv, "--weights", NULL);
        const int size = arg_int(argc, argv, "--size", 9);
        const int sims = arg_int(argc, argv, "--sims", 400);
        const double komi = arg_dbl(argc, argv, "--komi", 7.0);
        const char *sd = arg_str(argc, argv, "--seed", NULL);
        return gtp_main(w, sims, size, komi, sd ? strtoull(sd, NULL, 10) : 1234);
    }

    if (!strcmp(cmd, "play")) {
        const char *w = arg_str(argc, argv, "--weights", NULL);
        const int size = arg_int(argc, argv, "--size", 9);
        const int sims = arg_int(argc, argv, "--sims", 400);
        const double komi = arg_dbl(argc, argv, "--komi", 7.0);
        const int human = has_flag(argc, argv, "--white") ? 2 : 1;
        return human_play_main(w, sims, size, komi, human);
    }

    if (!strcmp(cmd, "selfplay")) {
        const char *w = arg_str(argc, argv, "--weights", NULL);
        const int games = arg_int(argc, argv, "--games", 5);
        const int sims = arg_int(argc, argv, "--sims", 200);
        const int size = arg_int(argc, argv, "--size", 9);
        const double komi = arg_dbl(argc, argv, "--komi", 7.0);
        const char *out = arg_str(argc, argv, "--out", "runs/games");
        const char *sd = arg_str(argc, argv, "--seed", NULL);
        return selfplay_main(w, games, sims, size, komi, out, sd ? strtoull(sd, NULL, 10) : 999);
    }

    if (!strcmp(cmd, "eval")) {
        const char *a = arg_str(argc, argv, "--a", "random");
        const char *b = arg_str(argc, argv, "--b", "random");
        const int games = arg_int(argc, argv, "--games", 20);
        const int sims = arg_int(argc, argv, "--sims", 100);
        const int size = arg_int(argc, argv, "--size", 9);
        const double komi = arg_dbl(argc, argv, "--komi", 7.0);
        const char *sd = arg_str(argc, argv, "--seed", NULL);
        const int open_plies = arg_int(argc, argv, "--open-plies", 4);
        { const char *os = arg_str(argc, argv, "--open-seed", NULL);
          if (os) eval_set_open_seed(strtoull(os, NULL, 10)); }
        return eval_main(a, b, games, sims, size, open_plies, komi, sd ? strtoull(sd, NULL, 10) : 4242);
    }

    usage();
    return 1;
}
