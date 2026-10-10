/* goboard.c - 本地人机对弈（AI 权重可直接编译进程序）
 *
 * 两个界面，编译时二选一：
 *   · 默认              ncurses 全屏棋盘（macOS / Linux，方向键操作）
 *   · -DGOAI_NO_CURSES  纯文本界面（Windows 无需装任何库，输入坐标落子）
 *
 * 权重来源（--weights）：
 *   builtin  用编译进程序的权重（默认，单个可执行文件即可运行，方便发给别人）
 *   auto     自动挑选最新的 runs_live/best.bin 等文件（边训练边下棋用）
 *   路径     指定 .bin 文件
 *
 * 编译：make goboard                     运行：./build/GoBoard --sims 200
 * Windows：mingw32-make -f Makefile.win goboard
 */
#include <ctype.h>
#include <math.h>
#ifndef GOAI_NO_CURSES
#  include <pthread.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "board.h"
#include "compat.h"
#include "mcts.h"
#include "net.h"
#include "rand.h"
#include "weights_builtin.h"

#ifndef GOAI_NO_CURSES
#  include <curses.h>
#  include <locale.h>
#endif

#define MAX_HIST 1024
#define MAX_SGF  8192

enum { ACT_NONE = 0, ACT_MOVE, ACT_PASS, ACT_UNDO, ACT_HINT, ACT_SAVE, ACT_NEW, ACT_RESIGN, ACT_QUIT };

typedef struct {
    int size;
    double komi;
    int sims;
    int human_color;
    int handicap;
    char weights[1024];
    char games_dir[512];
    int selftest;
} Options;

typedef struct {
    Options opt;
    Board   board;
    Board   hist[MAX_HIST];
    int     nhist;
    int     last_move;
    int     move_no;
    char    sgf[MAX_SGF];
    size_t  sgf_len;
    Net     net;
    Search  search;
    int     have_net;
    char    loaded_path[256];
    time_t  loaded_mtime;
    int     quit;
    int     hint_move;
    char    message[256];
    int     msg_ticks;
} App;

typedef struct { int action; int x; int y; } UiInput;

typedef struct {
    App   *app;
    Board  board;
    int    sims;
    volatile int progress;
    volatile int done;
    int    move;
    float  value;
} AIJob;

/* ------------------------------------------------------------------ 工具 */

static const char *pick_file(App *a, char *buf, size_t n) {
    (void)a;
    const char *cands[] = { "runs9_s800/best.bin", "runs9_s800/latest.bin",
                            "runs_step5/latest.bin", "runs_v4/latest.bin",
                            "runs_live/best.bin", "runs_live/latest.bin",
                            "runs_gpu/best.bin", "runs_gpu/latest.bin",
                            "runs_v2/latest.bin", "runs/latest.bin",
                            "versions/goai9x9_v5_800sims.bin", "weights.bin", NULL };
    const char *best = NULL;
    time_t best_t = 0;
    for (int i = 0; cands[i]; i++) {
        struct stat st;
        if (stat(cands[i], &st) == 0 && st.st_size > 0 && st.st_mtime >= best_t) {
            best_t = st.st_mtime;
            best = cands[i];
        }
    }
    if (!best) return NULL;
    snprintf(buf, n, "%s", best);
    return buf;
}

static int load_builtin(App *a) {
    if (!net_load_mem(&a->net, GOAI_BUILTIN_NET, GOAI_BUILTIN_NET_LEN)) return 0;
    search_set_net(&a->search, &a->net);
    a->search.komi = a->opt.komi;
    a->have_net = 1;
    a->loaded_mtime = 0;
    char arch[64];
    net_arch_str(&a->net, arch, sizeof(arch));
    snprintf(a->loaded_path, sizeof(a->loaded_path), "内置权重(%s)", arch);
    return 1;
}

static int maybe_reload(App *a) {
    if (strcmp(a->opt.weights, "auto") != 0 && strcmp(a->opt.weights, "builtin") != 0) {
        struct stat st;
        if (stat(a->opt.weights, &st) != 0) return 0;
        if (a->have_net && !strcmp(a->loaded_path, a->opt.weights) && st.st_mtime == a->loaded_mtime)
            return 0;
        if (!net_load(&a->net, a->opt.weights)) return 0;
        search_set_net(&a->search, &a->net);
        a->search.komi = a->opt.komi;
        a->have_net = 1;
        a->loaded_mtime = st.st_mtime;
        snprintf(a->loaded_path, sizeof(a->loaded_path), "%s", a->opt.weights);
        return 1;
    }
    if (!strcmp(a->opt.weights, "builtin")) {
        return a->have_net ? 0 : load_builtin(a);
    }
    /* auto：有训练输出就用最新的，没有就退回内置权重 */
    char path[1024];
    if (!pick_file(a, path, sizeof(path))) return a->have_net ? 0 : load_builtin(a);
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (a->have_net && !strcmp(path, a->loaded_path) && st.st_mtime == a->loaded_mtime) return 0;
    if (!net_load(&a->net, path)) return 0;
    search_set_net(&a->search, &a->net);
    a->search.komi = a->opt.komi;
    a->have_net = 1;
    a->loaded_mtime = st.st_mtime;
    snprintf(a->loaded_path, sizeof(a->loaded_path), "%s", path);
    snprintf(a->message, sizeof(a->message), "已热加载网络 %s", path);
    a->msg_ticks = 30;
    return 1;
}

static void set_msg(App *a, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(a->message, sizeof(a->message), fmt, ap);
    va_end(ap);
    a->msg_ticks = 30;
}

static void sgf_add(App *a, int move, int color) {
    char buf[32];
    if (move == M_PASS) snprintf(buf, sizeof(buf), ";%c[]", color == 1 ? 'B' : 'W');
    else snprintf(buf, sizeof(buf), ";%c[%c%c]", color == 1 ? 'B' : 'W',
                  'a' + (move % a->opt.size), 'a' + (move / a->opt.size));
    const size_t l = strlen(buf);
    if (a->sgf_len + l + 1 < MAX_SGF) {
        memcpy(a->sgf + a->sgf_len, buf, l);
        a->sgf_len += l;
        a->sgf[a->sgf_len] = 0;
    }
}

static void sgf_save(App *a, int winner) {
    goai_mkdir_p(a->opt.games_dir);
    char name[640], stamp[32];
    goai_localtime_str(stamp, sizeof(stamp), "%Y%m%d_%H%M%S");
    snprintf(name, sizeof(name), "%s/%s_%dx%d.sgf", a->opt.games_dir, stamp, a->opt.size, a->opt.size);
    FILE *f = fopen(name, "w");
    if (!f) { set_msg(a, "棋谱保存失败"); return; }
    fprintf(f, "(;GM[1]FF[4]CA[UTF-8]AP[GoBoard:2.0]SZ[%d]KM[%.1f]PB[%s]PW[%s]RU[Chinese]",
            a->opt.size, a->opt.komi,
            a->opt.human_color == 1 ? "Human" : "GoAI",
            a->opt.human_color == 1 ? "GoAI" : "Human");
    fprintf(f, "RE[%s]", winner == 1 ? "B+R" : (winner == 2 ? "W+R" : "0"));
    fputs(a->sgf, f);
    fputs(")\n", f);
    fclose(f);
    set_msg(a, "棋谱已保存：%s", name);
}

static const char *coord_str(int size, int move, char *buf, size_t n) {
    if (move == M_PASS) { snprintf(buf, n, "停着"); return buf; }
    if (move < 0) { snprintf(buf, n, "--"); return buf; }
    const int x = move % size, y = move / size;
    snprintf(buf, n, "%c%d", x < 8 ? 'A' + x : 'A' + x + 1, size - y);
    return buf;
}

/* --------------------------------------------------------------- 界面：ncurses */
#ifndef GOAI_NO_CURSES

static int col_letter(int x) { return x < 8 ? 'A' + x : 'A' + x + 1; }

static void draw_board(App *a, int cx, int cy) {
    const int n = a->opt.size;
    move(0, 0);
    attron(A_BOLD);
    printw("  GoBoard - %dx%d - %s - %s", n, n, a->opt.human_color == 1 ? "你执黑" : "你执白", a->loaded_path);
    attroff(A_BOLD);
    printw("\n\n   ");
    for (int x = 0; x < n; x++) printw("%c ", col_letter(x));
    printw("\n");
    for (int y = 0; y < n; y++) {
        printw("%2d ", n - y);
        for (int x = 0; x < n; x++) {
            const int p = y * n + x;
            const int8_t v = a->board.cell[p];
            const int cursor = (x == cx && y == cy);
            int attr = 0, pair = 0;
            char ch = '.';
            if (v == 1) { ch = 'X'; pair = 2; }
            else if (v == 2) { ch = 'O'; pair = 3; }
            if (p == a->last_move) attr |= A_BOLD;
            if (p == a->hint_move) attr |= A_UNDERLINE;
            if (cursor) attr |= A_REVERSE;
            if (pair) attron(COLOR_PAIR(pair));
            if (attr) attron(attr);
            printw("%c ", ch);
            if (attr) attroff(attr);
            if (pair) attroff(COLOR_PAIR(pair));
        }
        printw("%2d\n", n - y);
    }
    printw("   ");
    for (int x = 0; x < n; x++) printw("%c ", col_letter(x));
    printw("\n");
}

static void draw_status(App *a) {
    printw("\n  ------------------------------------------------\n");
    const int turn = a->board.to_move;
    if (turn == a->opt.human_color) { attron(A_BOLD); printw("  轮到你落子（%s）\n", turn == 1 ? "黑" : "白"); attroff(A_BOLD); }
    else printw("  轮到 AI…\n");
    printw("  提子：你 %d / AI %d      手数 %d\n",
           a->board.captures[a->opt.human_color], a->board.captures[3 - a->opt.human_color], a->move_no);
    if (a->msg_ticks > 0) printw("  >> %s\n", a->message);
    printw("\n  方向键/hjkl 移动 - 回车落子 - p 停着 - u 悔棋 - i 提示 - s 存棋谱 - n 新局 - r 认输 - q 退出\n");
}

static void ui_init(App *a) { (void)a; setlocale(LC_ALL, ""); initscr(); cbreak(); noecho(); keypad(stdscr, TRUE); curs_set(0);
    if (has_colors()) { start_color(); use_default_colors(); init_pair(2, COLOR_WHITE, COLOR_BLACK); init_pair(3, COLOR_BLACK, COLOR_WHITE); } }
static void ui_shutdown(void) { endwin(); }

static void ui_render(App *a, int *cx, int *cy) { erase(); draw_board(a, *cx, *cy); draw_status(a); refresh(); }

static void ui_progress(App *a, int *cx, int *cy, int done, int total) {
    erase(); draw_board(a, *cx, *cy);
    printw("\n  ------------------------------------------------\n");
    const int w = 28, fill = total > 0 ? done * w / total : 0;
    printw("  AI 思考中  [");
    for (int i = 0; i < w; i++) printw(i < fill ? "#" : ".");
    printw("] %d/%d\n", done, total);
    refresh();
}

static void ui_result(App *a, int winner, double score) {
    erase();
    int cx = -1, cy = -1;
    draw_board(a, cx, cy);
    printw("\n  ================================================\n");
    if (winner == 0) printw("  终局：和棋（%.1f 目）\n", score);
    else printw("  终局：%s 胜 %.1f 目 -- %s\n", winner == 1 ? "黑" : "白", fabs(score),
                winner == a->opt.human_color ? "你赢了！" : "AI 赢了");
    printw("  提子：你 %d / AI %d · 共 %d 手\n",
           a->board.captures[a->opt.human_color], a->board.captures[3 - a->opt.human_color], a->move_no);
    if (a->msg_ticks > 0) printw("  >> %s\n", a->message);
    printw("\n  按 n 再来一局，按 q 退出\n");
    refresh();
}

static UiInput ui_input(App *a, int *cx, int *cy) {
    UiInput in = { ACT_NONE, *cx, *cy };
    const int ch = getch();
    switch (ch) {
        case KEY_UP:    case 'k': if (*cy > 0) (*cy)--; break;
        case KEY_DOWN:  case 'j': if (*cy < a->opt.size - 1) (*cy)++; break;
        case KEY_LEFT:  case 'h': if (*cx > 0) (*cx)--; break;
        case KEY_RIGHT: case 'l': if (*cx < a->opt.size - 1) (*cx)++; break;
        case '\n': case KEY_ENTER: case ' ': in.action = ACT_MOVE; break;
        case 'p': in.action = ACT_PASS; break;
        case 'u': in.action = ACT_UNDO; break;
        case 'i': in.action = ACT_HINT; break;
        case 's': in.action = ACT_SAVE; break;
        case 'n': in.action = ACT_NEW; break;
        case 'r': in.action = ACT_RESIGN; break;
        case 'q': in.action = ACT_QUIT; break;
        default: break;
    }
    in.x = *cx; in.y = *cy;
    return in;
}

static int ui_confirm_new_game(void) { int ch; do { ch = getch(); } while (ch != 'n' && ch != 'q'); return ch; }

/* --------------------------------------------------------------- 界面：文本 */
#else

static void print_board(App *a) {
    const int n = a->opt.size;
    printf("\n    ");
    for (int x = 0; x < n; x++) printf("%c ", x < 8 ? 'A' + x : 'A' + x + 1);
    printf("\n");
    for (int y = 0; y < n; y++) {
        printf(" %2d ", n - y);
        for (int x = 0; x < n; x++) {
            const int p = y * n + x;
            const int8_t v = a->board.cell[p];
            char ch = '.';
            if (v == 1) ch = 'X';
            else if (v == 2) ch = 'O';
            if (p == a->last_move) printf("[%c]", ch);
            else if (p == a->hint_move) printf("<%c>", ch);
            else printf(" %c ", ch);
        }
        printf("%2d\n", n - y);
    }
    printf("    ");
    for (int x = 0; x < n; x++) printf("%c ", x < 8 ? 'A' + x : 'A' + x + 1);
    printf("\n");
}

static void ui_init(App *a) {
    printf("\n================ GoBoard %dx%d ================\n", a->opt.size, a->opt.size);
    printf("  黑棋 X · 白棋 O · 你执%s · AI 每步搜索 %d 次 · 贴目 %.1f\n",
           a->opt.human_color == 1 ? "黑（先行）" : "白", a->opt.sims, a->opt.komi);
    printf("  输入坐标落子（如 D4），或 pass/undo/hint/save/new/resign/quit\n");
}
static void ui_shutdown(void) { printf("\n再见。\n"); }

static void ui_render(App *a, int *cx, int *cy) {
    (void)cx; (void)cy;
    print_board(a);
    printf("  ------------------------------------------------------------\n");
    printf("  你执%s · 提子 你 %d / AI %d · 手数 %d · 网络 %s\n",
           a->opt.human_color == 1 ? "黑" : "白",
           a->board.captures[a->opt.human_color], a->board.captures[3 - a->opt.human_color],
           a->move_no, a->loaded_path);
    if (a->msg_ticks > 0) printf("  >> %s\n", a->message);
}
static void ui_result(App *a, int winner, double score) {
    print_board(a);
    printf("  ============================================================\n");
    if (winner == 0) printf("  终局：和棋（%.1f 目）\n", score);
    else printf("  终局：%s 胜 %.1f 目 -- %s\n", winner == 1 ? "黑" : "白", fabs(score),
                winner == a->opt.human_color ? "你赢了！" : "AI 赢了");
    printf("  提子：你 %d / AI %d · 共 %d 手\n",
           a->board.captures[a->opt.human_color], a->board.captures[3 - a->opt.human_color], a->move_no);
    if (a->msg_ticks > 0) printf("  >> %s\n", a->message);
}
static int ui_confirm_new_game(void) {
    char buf[32];
    printf("\n  再来一局？(y/n) ");
    fflush(stdout);
    if (!fgets(buf, sizeof(buf), stdin)) return 'q';
    return (buf[0] == 'y' || buf[0] == 'Y') ? 'n' : 'q';
}

static UiInput ui_input(App *a, int *cx, int *cy) {
    (void)cx; (void)cy;
    UiInput in = { ACT_NONE, 0, 0 };
    char buf[128];
    printf("\n  落子 > ");
    fflush(stdout);
    if (!fgets(buf, sizeof(buf), stdin)) { in.action = ACT_QUIT; return in; }
    for (char *p = buf; *p; p++) *p = (char)tolower((unsigned char)*p);
    if (buf[0] == 'p' || strncmp(buf, "pass", 4) == 0) { in.action = ACT_PASS; return in; }
    if (buf[0] == 'u' || strncmp(buf, "undo", 4) == 0) { in.action = ACT_UNDO; return in; }
    if (strncmp(buf, "hint", 4) == 0) { in.action = ACT_HINT; return in; }
    if (strncmp(buf, "save", 4) == 0) { in.action = ACT_SAVE; return in; }
    if (strncmp(buf, "new", 3) == 0) { in.action = ACT_NEW; return in; }
    if (strncmp(buf, "resign", 6) == 0) { in.action = ACT_RESIGN; return in; }
    if (buf[0] == 'q' || strncmp(buf, "quit", 4) == 0) { in.action = ACT_QUIT; return in; }
    char c = 0;
    int row = 0;
    if (sscanf(buf, " %c%d", &c, &row) == 2) {
        int col = -1;
        if (c >= 'a' && c <= 'h') col = c - 'a';
        else if (c >= 'i' && c <= 'z') col = c - 'a' - 1;   /* 跳过 I */
        if (col >= 0 && col < a->opt.size && row >= 1 && row <= a->opt.size) {
            in.action = ACT_MOVE;
            in.x = col;
            in.y = a->opt.size - row;
            return in;
        }
    }
    printf("  看不懂这个输入。例子：D4 / pass / undo / hint / save / new / resign / quit\n");
    return in;
}
#endif /* GOAI_NO_CURSES */

/* ------------------------------------------------------------------ AI */

#ifndef GOAI_NO_CURSES
static void *ai_thread(void *arg) {
    AIJob *j = (AIJob *)arg;
    App *a = j->app;
    float policy[BOARD_MAX_POINTS + 1];
    a->search.progress_done = &j->progress;
    a->search.progress_total = j->sims;
    j->move = search_run(&a->search, &j->board, j->sims, 0.0f, 0.0f, 0.0f, policy, &j->value);
    a->search.progress_done = NULL;
    j->done = 1;
    return NULL;
}
#endif

static int game_over(const App *a) {
    return a->board.passes >= 2 || a->board.nmoves >= 3 * a->opt.size * a->opt.size;
}

static void run_ai_move(App *a, int *cx, int *cy) {
    maybe_reload(a);
    AIJob j;
    memset(&j, 0, sizeof(j));
    j.app = a;
    j.board = a->board;
    j.sims = a->opt.sims;
#ifdef GOAI_NO_CURSES
    /* 文本版：同步搜索（Windows 上不需要线程库，少一个依赖） */
    float policy[BOARD_MAX_POINTS + 1];
    printf("  AI 思考中…（%d 次模拟）\n", j.sims);
    fflush(stdout);
    j.move = search_run(&a->search, &j.board, j.sims, 0.0f, 0.0f, 0.0f, policy, &j.value);
    (void)cx; (void)cy;
#else
    /* 全屏版：放到后台线程，主线程刷新进度条 */
    pthread_t th;
    pthread_create(&th, NULL, ai_thread, &j);
    while (!j.done) {
        ui_progress(a, cx, cy, j.progress, j.sims);
        goai_sleep_ms(60);
    }
    pthread_join(th, NULL);
#endif
    a->hist[a->nhist < MAX_HIST ? a->nhist : MAX_HIST - 1] = a->board;
    if (a->nhist < MAX_HIST) a->nhist++;
    const int color = a->board.to_move;
    if (!board_play(&a->board, j.move)) board_play(&a->board, M_PASS);
    sgf_add(a, j.move, color);
    a->last_move = j.move;
    a->move_no++;
    char v[16];
    coord_str(a->opt.size, j.move, v, sizeof(v));
    set_msg(a, "AI 落子 %s（胜率估计 %.0f%%）", v, 50.0 * (1.0 + j.value));
}

static void human_move(App *a, int x, int y) {
    const int p = y * a->opt.size + x;
    if (p < 0 || p >= a->opt.size * a->opt.size) return;
    if (a->board.cell[p] != 0) { set_msg(a, "这个点已经有子了"); return; }
    if (!board_is_legal_move(&a->board, p)) { set_msg(a, "这里不能下（禁入点或劫）"); return; }
    a->hist[a->nhist < MAX_HIST ? a->nhist : MAX_HIST - 1] = a->board;
    if (a->nhist < MAX_HIST) a->nhist++;
    const int color = a->board.to_move;
    board_play(&a->board, p);
    sgf_add(a, p, color);
    a->last_move = p;
    a->move_no++;
    a->hint_move = -1;
    search_begin_move(&a->search);
    set_msg(a, "你落子完成");
}

static void do_pass(App *a) {
    const int color = a->board.to_move;
    a->hist[a->nhist < MAX_HIST ? a->nhist : MAX_HIST - 1] = a->board;
    if (a->nhist < MAX_HIST) a->nhist++;
    board_play(&a->board, M_PASS);
    sgf_add(a, M_PASS, color);
    a->move_no++;
    set_msg(a, "你选择了停着");
}

static void undo(App *a) {
    int steps = 0;
    while (a->nhist > 0 && steps < 2) {
        a->board = a->hist[--a->nhist];
        steps++;
        if (a->board.to_move == a->opt.human_color) break;
    }
    if (a->sgf_len > 0) {
        size_t cut = a->sgf_len;
        int semis = 0;
        for (size_t i = a->sgf_len; i > 0; i--) {
            if (a->sgf[i - 1] == ';' && ++semis == steps) { cut = i - 1; break; }
        }
        a->sgf_len = cut;
        a->sgf[cut] = 0;
    }
    a->move_no = a->move_no > steps ? a->move_no - steps : 0;
    a->last_move = -1;
    search_begin_move(&a->search);
    set_msg(a, "已悔棋 %d 手", steps);
}

static void show_hint(App *a) {
    if (a->board.to_move != a->opt.human_color) { set_msg(a, "现在不是你的回合"); return; }
    maybe_reload(a);
    Board tmp = a->board;
    float pol[BOARD_MAX_POINTS + 1], val = 0;
    const int sims = a->opt.sims / 4 > 0 ? a->opt.sims / 4 : 20;
    const int mv = search_run(&a->search, &tmp, sims, 0.0f, 0.0f, 0.0f, pol, &val);
    a->hint_move = mv;
    char v[16];
    coord_str(a->opt.size, mv, v, sizeof(v));
    set_msg(a, "建议下在 %s（该局面胜率估计 %.0f%%）", v, 50.0 * (1.0 + val));
    search_begin_move(&a->search);
}

static void new_game(App *a) {
    board_init(&a->board, a->opt.size);
    a->nhist = 0;
    a->last_move = -1;
    a->move_no = 0;
    a->sgf_len = 0;
    a->sgf[0] = 0;
    a->hint_move = -1;
    if (a->opt.handicap > 0 && a->opt.human_color == 1) {
        const int n = a->opt.size;
        const int pts[9][2] = {{3,3},{n-4,3},{3,n-4},{n-4,n-4},{n/2,n/2},
                               {3,n/2},{n-4,n/2},{n/2,3},{n/2,n-4}};
        for (int i = 0; i < a->opt.handicap && i < 9; i++) {
            a->board.cell[pts[i][1] * n + pts[i][0]] = 1;
            sgf_add(a, pts[i][1] * n + pts[i][0], 1);
        }
    }
    search_begin_move(&a->search);
}

static int selftest(App *a) {
    printf("自检：AI 自我对弈 40 手（%dx%d，每步 %d 次模拟，权重 %s）\n",
           a->opt.size, a->opt.size, a->opt.sims, a->loaded_path);
    for (int i = 0; i < 40 && !game_over(a); i++) {
        float pol[BOARD_MAX_POINTS + 1], val = 0;
        Board tmp = a->board;
        const int mv = search_run(&a->search, &tmp, a->opt.sims, 0.0f, 0.0f, 0.0f, pol, &val);
        char v[16];
        coord_str(a->opt.size, mv, v, sizeof(v));
        printf("  %2d. %s %s  胜率 %.0f%%\n", i + 1, a->board.to_move == 1 ? "黑" : "白", v, 50.0 * (1.0 + val));
        if (!board_play(&a->board, mv)) board_play(&a->board, M_PASS);
        search_begin_move(&a->search);
    }
    printf("终局分数（黑-白-贴目）= %.1f\n", board_score(&a->board, a->opt.komi));
    return 0;
}

int main(int argc, char **argv) {
    App *a = (App *)calloc(1, sizeof(App));
    a->opt.size = 9;
    a->opt.komi = -1;
    a->opt.sims = 200;
    a->opt.human_color = 1;
    snprintf(a->opt.weights, sizeof(a->opt.weights), "builtin");
    snprintf(a->opt.games_dir, sizeof(a->opt.games_dir), "games");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--size") && i + 1 < argc) a->opt.size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--komi") && i + 1 < argc) a->opt.komi = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sims") && i + 1 < argc) a->opt.sims = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) snprintf(a->opt.weights, sizeof(a->opt.weights), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--white")) a->opt.human_color = 2;
        else if (!strcmp(argv[i], "--black")) a->opt.human_color = 1;
        else if (!strcmp(argv[i], "--handicap") && i + 1 < argc) a->opt.handicap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) snprintf(a->opt.games_dir, sizeof(a->opt.games_dir), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--selftest")) a->opt.selftest = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("GoBoard - 本地人机对战（AI 已在程序内）\n"
                   "  --size N      棋盘 9/13/19（默认 9）\n"
                   "  --sims N      AI 每步搜索次数（默认 200，越大越强越慢）\n"
                   "  --komi X      贴目（默认 9 路 7.0，19 路 7.5）\n"
                   "  --weights S   builtin（内置，默认）/ auto（有训练输出就用最新的）/ 文件路径\n"
                   "  --white       你执白（默认执黑先行）\n"
                   "  --handicap N  让子数\n"
                   "  --selftest    自检：AI 自对弈打印过程，不用交互\n");
            return 0;
        }
    }
    if (a->opt.komi < 0) a->opt.komi = (a->opt.size >= 19) ? 7.5 : 7.0;
    if (a->opt.size < 5 || a->opt.size > BOARD_MAX) { fprintf(stderr, "棋盘大小需在 5..%d\n", BOARD_MAX); return 1; }

    board_init(&a->board, a->opt.size);
    search_init(&a->search, a->opt.size, NULL, 20241001ULL, a->opt.komi);
    a->search.pass_min_move = 0;
    a->search.max_moves = 3 * a->opt.size * a->opt.size;
    a->hint_move = -1;
    a->last_move = -1;
    if (!maybe_reload(a)) { fprintf(stderr, "权重加载失败：%s\n", a->opt.weights); return 1; }

    if (a->opt.selftest) {
        const int rc = selftest(a);
        search_free(&a->search);
        net_free(&a->net);
        free(a);
        return rc;
    }

    ui_init(a);
    new_game(a);
    int cx = a->opt.size / 2, cy = a->opt.size / 2;
    while (!a->quit) {
        if (game_over(a)) {
            const double sc = board_score(&a->board, a->opt.komi);
            const int winner = sc > 0 ? 1 : (sc < 0 ? 2 : 0);
            if (winner != 0) sgf_save(a, winner);
            ui_result(a, winner, sc);
            if (ui_confirm_new_game() == 'q') break;
            new_game(a);
            continue;
        }
        if (a->board.to_move != a->opt.human_color) { run_ai_move(a, &cx, &cy); continue; }
        ui_render(a, &cx, &cy);
        const UiInput in = ui_input(a, &cx, &cy);
        if (a->msg_ticks > 0) a->msg_ticks--;
        switch (in.action) {
            case ACT_MOVE:   human_move(a, in.x, in.y); break;
            case ACT_PASS:   do_pass(a); break;
            case ACT_UNDO:   undo(a); break;
            case ACT_HINT:   show_hint(a); break;
            case ACT_SAVE: { const double sc = board_score(&a->board, a->opt.komi);
                             sgf_save(a, sc > 0 ? 1 : (sc < 0 ? 2 : 0)); break; }
            case ACT_NEW:    new_game(a); break;
            case ACT_RESIGN: sgf_save(a, a->opt.human_color == 1 ? 2 : 1); a->quit = 1; break;
            case ACT_QUIT:   a->quit = 1; break;
            default: break;
        }
    }
    ui_shutdown();
    search_free(&a->search);
    net_free(&a->net);
    free(a);
    return 0;
}
