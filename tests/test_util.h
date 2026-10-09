#ifndef GOAI_TEST_UTIL_H
#define GOAI_TEST_UTIL_H
#include <stdio.h>
extern int g_checks, g_failures;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        g_checks++;                                                             \
        if (!(cond)) {                                                          \
            g_failures++;                                                       \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                       \
            printf(__VA_ARGS__);                                                \
            printf("\n");                                                      \
        }                                                                       \
    } while (0)
#endif
