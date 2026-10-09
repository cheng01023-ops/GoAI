/* apps.c - GTP engine, terminal play, SGF self-play export and match evaluation */
#include "apps.h"
#include "compat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------- helpers */

typedef struct {
    Net      net;
    NetCache cache;
    int      loaded;
    int      size;
} NetHandle;

static int net_handle_load(NetHandle *h, const char *path, int size) {
    if (!path) return 0;
    int planes, channels, vhidden;
    net_config_default(size, &planes, &channels, &vhidden);
    net_init(&h->net, size, planes, channels, vhidden, 1);
    if (!net_load(&h->net, path)) {
        fprintf(stderr, "error: cannot read weights '%s'\n", path);
        net_free(&h->net);
        return -1;
    }
    h->size = h->net.size;
    net_cache_init(&h->net, &h->cache);
    h->loaded = 1;
    return 1;
}

static void net_handle_free(NetHandle *h) {
    if (h->loaded) { net_cache_free(&h->cache); net_free(&h->net); }
    h->loaded = 0;
}

static void print_board(const Board *b) {
    char buf[4096];
    board_to_string(b, buf, sizeof(buf));
    fputs(buf, stdout);
}

/* ------------------------------------------------------------------ GTP */

static void gtp_respond(const char *s) { printf("= %s\n\n", s ? s : ""); fflush(stdout); }
static void gtp_error(const char *s)    { printf("? %s\n\n", s ? s : ""); fflush(stdout); }

static void gtp_parse_args(char *line, char **argv, int *argc, int max) {
    int n = 0;
    char *p = line;
    while (*p && n < max) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        argv[n++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        if (*p) *p++ = 0;
    }
    *argc = n;
}

int gtp_main(const char *weights, int sims, int size, double komi, uint64_t seed) {
    NetHandle h;
    memset(&h, 0, sizeof(h));
    if (net_handle_load(&h, weights, size) < 0) return 1;

    Search s;
    search_init(&s, size, h.loaded ? &h.net : NULL, seed, komi);
    s.pass_min_move = 0;                       /* a player may always pass in GTP */
    s.max_moves = 3 * size * size;

    Board b;
    board_init(&b, size);
    Board history[1024];
    int nhist = 0;

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (line[0] == 0) continue;
        char *tok[16];
        int ntok = 0;
        gtp_parse_args(line, tok, &ntok, 16);
        if (ntok == 0) continue;
        const char *cmd = tok[0];

        if (!strcmp(cmd, "protocol_version")) { gtp_respond("2"); }
        else if (!strcmp(cmd, "name"))   { gtp_respond("GoAI"); }
        else if (!strcmp(cmd, "version")){ gtp_respond("1.0"); }
        else if (!strcmp(cmd, "list_commands")) {
            gtp_respond("protocol_version\nname\nversion\nlist_commands\nboardsize\n"
                        "clear_board\nkomi\nplay\ngenmove\nundo\nshowboard\nfinal_score\nquit");
        }
        else if (!strcmp(cmd, "boardsize")) {
            const int n = ntok > 1 ? atoi(tok[1]) : 0;
            if (n < 2 || n > BOARD_MAX) gtp_error("unacceptable size");
            else if (h.loaded && n != h.size) gtp_error("weights are for another board size");
            else {
                board_init(&b, n);
                search_free(&s);
                search_init(&s, n, h.loaded ? &h.net : NULL, seed, komi);
                s.pass_min_move = 0;
                s.max_moves = 3 * n * n;
                nhist = 0;
                gtp_respond("");
            }
        }
        else if (!strcmp(cmd, "clear_board")) {
            board_init(&b, b.size);
            nhist = 0;
            gtp_respond("");
        }
        else if (!strcmp(cmd, "komi")) { komi = ntok > 1 ? atof(tok[1]) : komi; s.komi = komi; gtp_respond(""); }
        else if (!strcmp(cmd, "play") || !strcmp(cmd, "genmove")) {
            const int is_play = !strcmp(cmd, "play");
            if ((is_play && ntok < 3) || ntok < 2) { gtp_error("syntax error"); continue; }
            char c = tok[1][0];
            const int color = (c == 'b' || c == 'B') ? 1 : 2;
            if (is_play) {
                const int mv = board_parse_vertex(&b, tok[2]);
                if (mv == M_NONE) { gtp_error("invalid vertex"); continue; }
                history[nhist < 1024 ? nhist : 1023] = b;
                if (nhist < 1024) nhist++;
                b.to_move = color;
                if (!board_play(&b, mv)) { nhist--; gtp_error("illegal move"); continue; }
                gtp_respond("");
            } else {
                b.to_move = color;
                float pol[BOARD_MAX_POINTS + 1], rv = 0;
                const int mv = search_run(&s, &b, sims, 0.0f, 0.0f, 0.0f, pol, &rv);
                char v[16];
                board_vertex(&b, mv, v, sizeof(v));
                history[nhist < 1024 ? nhist : 1023] = b;
                if (nhist < 1024) nhist++;
                if (!board_play(&b, mv)) { nhist--; gtp_error("engine produced an illegal move"); continue; }
                char out[128];
                snprintf(out, sizeof(out), "%s (win%% %.1f)", v, 50.0 * (1.0 + rv));
                gtp_respond(out);
            }
        }
        else if (!strcmp(cmd, "undo")) {
            if (nhist <= 0) gtp_error("cannot undo");
            else { b = history[--nhist]; gtp_respond(""); }
        }
        else if (!strcmp(cmd, "showboard")) {
            printf("= ");
            print_board(&b);
            printf("\n");
            fflush(stdout);
        }
        else if (!strcmp(cmd, "final_score")) {
            const double sc = board_score(&b, komi);
            char out[64];
            if (fabs(sc) < 1e-9) snprintf(out, sizeof(out), "0");
            else snprintf(out, sizeof(out), "%c+%.1f", sc > 0 ? 'B' : 'W', fabs(sc));
            gtp_respond(out);
        }
        else if (!strcmp(cmd, "quit")) { gtp_respond(""); break; }
        else if (!strcmp(cmd, "kgs-game_over") || !strcmp(cmd, "time_left") ||
                 !strcmp(cmd, "final_status_list")) { gtp_respond(""); }
        else gtp_error("unknown command");
    }
    search_free(&s);
    net_handle_free(&h);
    return 0;
}

/* ---------------------------------------------------------- human vs AI */

int human_play_main(const char *weights, int sims, int size, double komi, int human_color) {
    NetHandle h;
    memset(&h, 0, sizeof(h));
    if (net_handle_load(&h, weights, size) < 0) return 1;
    Search s;
    search_init(&s, size, h.loaded ? &h.net : NULL, 12345, komi);
    s.pass_min_move = 2 * size;
    s.max_moves = 3 * size * size;

    Board b;
    board_init(&b, size);
    Board history[1024];
    int nhist = 0;

    printf("GoAI - you play %s on a %dx%d board (komi %.1f)\n",
           human_color == 1 ? "black" : "white", size, size, komi);
    printf("enter moves like D4, or: pass / undo / board / quit\n\n");

    char line[256];
    while (b.passes < 2) {
        print_board(&b);
        if (b.to_move != human_color) {
            float pol[BOARD_MAX_POINTS + 1], rv = 0;
            history[nhist < 1024 ? nhist : 1023] = b;
            if (nhist < 1024) nhist++;
            const int mv = search_run(&s, &b, sims, 0.0f, 0.0f, 0.0f, pol, &rv);
            char v[16];
            board_vertex(&b, mv, v, sizeof(v));
            printf("GoAI plays %s   (win%% %.1f)\n", v, 50.0 * (1.0 + rv));
            if (!board_play(&b, mv)) board_play(&b, M_PASS);
            continue;
        }
        printf("your move (%s): ", b.to_move == 1 ? "black" : "white");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (!strcmp(line, "quit") || !strcmp(line, "q")) break;
        if (!strcmp(line, "board")) continue;
        if (!strcmp(line, "undo")) {
            if (nhist > 0) b = history[--nhist];
            if (nhist > 0) b = history[--nhist];
            continue;
        }
        const int mv = board_parse_vertex(&b, line);
        if (mv == M_NONE) { printf("  ?? use coordinates like D4, or pass\n"); continue; }
        history[nhist < 1024 ? nhist : 1023] = b;
        if (nhist < 1024) nhist++;
        if (!board_play(&b, mv)) { nhist--; printf("  illegal move\n"); continue; }
    }
    print_board(&b);
    const double sc = board_score(&b, komi);
    printf("game over: %s by %.1f\n", sc > 0 ? "black wins" : (sc < 0 ? "white wins" : "draw"), fabs(sc));
    search_free(&s);
    net_handle_free(&h);
    return 0;
}

/* ------------------------------------------------------- SGF self-play  */

int selfplay_main(const char *weights, int games, int sims, int size, double komi,
                  const char *out_dir, uint64_t seed) {
    NetHandle h;
    memset(&h, 0, sizeof(h));
    if (net_handle_load(&h, weights, size) < 0) return 1;
    Search s;
    search_init(&s, size, h.loaded ? &h.net : NULL, seed, komi);
    s.pass_min_move = 2 * size;
    s.max_moves = 3 * size * size;

    if (goai_mkdir_p(out_dir ? out_dir : "runs/games") != 0) fprintf(stderr, "warning: mkdir failed\n");

    int black_wins = 0, white_wins = 0, draws = 0;
    for (int g = 0; g < games; g++) {
        Board b;
        board_init(&b, size);
        char path[1024];
        snprintf(path, sizeof(path), "%s/game%03d.sgf", out_dir ? out_dir : "runs/games", g + 1);
        char cmt[128];
        snprintf(cmt, sizeof(cmt), "GoAI self-play game %d (sims=%d)", g + 1, sims);
        Sgf *sgf = sgf_open(path, size, komi, cmt);
        while (b.passes < 2 && b.nmoves < s.max_moves) {
            const int color = b.to_move;
            float pol[BOARD_MAX_POINTS + 1];
            const int mv = search_run(&s, &b, sims, 0.10f, 0.0f, b.nmoves < 10 ? 0.6f : 0.0f, pol, NULL);
            if (sgf) sgf_move(sgf, mv, color);
            if (!board_play(&b, mv)) board_play(&b, M_PASS);
        }
        const int w = board_winner(&b, komi);
        if (w == 1) black_wins++;
        else if (w == 2) white_wins++;
        else draws++;
        if (sgf) { sgf_result(sgf, w); sgf_close(sgf); }
        printf("game %2d/%d: %s by %.1f  (%d moves)  -> %s\n", g + 1, games,
               w == 1 ? "black" : (w == 2 ? "white" : "draw"),
               fabs(board_score(&b, komi)), b.nmoves, path);
        fflush(stdout);
    }
    printf("\n%d games: black %d, white %d, draw %d\n", games, black_wins, white_wins, draws);
    search_free(&s);
    net_handle_free(&h);
    return 0;
}

/* -------------------------------------------------------------- matches */

static Net *load_net_or_null(const char *spec, int size, NetHandle *h, int *is_random) {
    if (!spec || !strcmp(spec, "random") || !strcmp(spec, "rand")) { *is_random = 1; return NULL; }
    if (net_handle_load(h, spec, size) < 0) exit(1);
    *is_random = 0;
    return &h->net;
}

int eval_main(const char *a_spec, const char *b_spec, int games, int sims, int size,
              int open_plies, double komi, uint64_t seed) {
    NetHandle ha, hb;
    memset(&ha, 0, sizeof(ha));
    memset(&hb, 0, sizeof(hb));
    int a_random = 0, b_random = 0;
    Net *na = load_net_or_null(a_spec, size, &ha, &a_random);
    Net *nb = load_net_or_null(b_spec, size, &hb, &b_random);

    TrainConfig cfg;
    train_config_default(&cfg, size);
    cfg.komi = komi;
    cfg.eval_sims = sims;
    cfg.open_plies = open_plies;

    Search s;
    search_init(&s, size, na ? na : nb, seed, komi);
    s.pass_min_move = cfg.pass_min_move;
    s.max_moves = cfg.max_moves;

    EngineSpec A = { a_random ? NULL : na, sims, a_random };
    EngineSpec B = { b_random ? NULL : nb, sims, b_random };

    const double t0 = goai_now();
    int wa = 0, wb = 0, dr = 0;
    eval_match(&s, &cfg, A, B, games, &wa, &wb, &dr);
    const double el = goai_now() - t0;
    printf("A = %s%s\nB = %s%s\n", a_spec ? a_spec : "random", a_random ? " (random mover)" : "",
           b_spec ? b_spec : "random", b_random ? " (random mover)" : "");
    printf("%d games at %d simulations%s: A wins %d, B wins %d, draws %d  => A win rate %.1f%%\n",
           games, sims, open_plies > 0 ? " (paired openings)" : "", wa, wb, dr,
           100.0 * (wa + 0.5 * dr) / (double)games);
    printf("(%.1fs, %.2fs per game)\n", el, el / games);

    search_free(&s);
    net_handle_free(&ha);
    net_handle_free(&hb);
    return 0;
}

/* ------------------------------------------------------------ diagnostics */

int diag_main(const char *weights, int size) {
    NetHandle h;
    memset(&h, 0, sizeof(h));
    if (net_handle_load(&h, weights, size) < 0) return 1;
    NetCache c;
    net_cache_init(&h.net, &c);
    const int nn = h.net.nn, planes = h.net.planes;
    float *x = (float *)calloc((size_t)planes * nn, sizeof(float));
    float pol[BOARD_MAX_POINTS + 1], v;

    struct { const char *name; int kind; } cases[] = {
        { "black owns 4 columns (black to move)", 0 },
        { "white owns 4 columns (black to move)", 1 },
        { "empty board (black to move)",          2 },
        { "empty board (white to move - symmetric)", 3 },
        { "black owns 4 columns (white to move)", 4 },
    };
    printf("network: %s  (%d params, %d channels)\n\n", weights, h.net.n_params, h.net.channels);
    for (unsigned k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        Board b;
        board_init(&b, size);
        const int kind = cases[k].kind;
        const int col = (kind == 1) ? 2 : 1;   /* case 1: the columns belong to white */
        if (kind <= 1 || kind == 4) {
            for (int xx = 0; xx < 4; xx++)
                for (int yy = 0; yy < size; yy++) b.cell[yy * size + xx] = (int8_t)col;
            b.nmoves = 4 * size;
        }
        b.to_move = (kind >= 3) ? 2 : 1;   /* kind 3/4 都是轮到白棋 */
        net_features(&b, x);
        net_forward(&h.net, &c, x, pol, &v);
        char mv[8];
        int best = 0;
        for (int i = 0; i <= nn; i++) if (pol[i] > pol[best]) best = i;
        board_vertex(&b, board_move_from_index(&b, best), mv, sizeof(mv));
        printf("%-42s value %+.5f   top policy move %s (%.1f%%)  pass %.1f%%\n",
               cases[k].name, v, mv, 100.0 * pol[best], 100.0 * pol[nn]);
    }
    free(x);
    net_cache_free(&c);
    net_handle_free(&h);
    return 0;
}
