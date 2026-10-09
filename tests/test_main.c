#include <stdio.h>
int g_checks = 0, g_failures = 0;
int test_board_run(void);
int test_rng_run(void);
int test_net_run(void);
int test_mcts_run(void);
int test_train_run(void);
int main(void) {
    printf("== GoAI test suite ==\n");
    printf("[board]\n"); test_board_run();
    printf("[rng]\n");   test_rng_run();
    printf("[net]\n");   test_net_run();
    printf("[mcts]\n");  test_mcts_run();
    printf("[train]\n");  test_train_run();
    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures == 0) printf("ALL TESTS PASSED\n");
    return g_failures ? 1 : 0;
}
