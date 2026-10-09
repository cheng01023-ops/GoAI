/* compat.c - 跨平台小工具实现 */
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
double goai_now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

void goai_localtime_str(char *buf, size_t n, const char *fmt) {
    time_t now = time(NULL);
    struct tm tmv;
    localtime_s(&tmv, &now);
    strftime(buf, n, fmt, &tmv);
}

long goai_pid(void) { return (long)_getpid(); }

int goai_cpu_count(void) {
    DWORD c = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return c > 0 ? (int)c : 1;
}

void goai_sleep_ms(int ms) { Sleep((DWORD)(ms < 0 ? 0 : ms)); }

#else

double goai_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void goai_localtime_str(char *buf, size_t n, const char *fmt) {
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(buf, n, fmt, &tmv);
}

long goai_pid(void) { return (long)getpid(); }

int goai_cpu_count(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    return c > 0 ? (int)c : 1;
#else
    return 1;
#endif
}

void goai_sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

#endif /* _WIN32 */


int goai_mkdir_p(const char *path) {
    if (!path || !*path) return -1;
    char tmp[1024];
    size_t len = strlen(path);
    if (len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i <= len; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\' || tmp[i] == 0) {
            const char c = tmp[i];
            tmp[i] = 0;
            GOAI_MKDIR_ONE(tmp);          /* 已存在时返回 -1，忽略即可 */
            tmp[i] = c;
            if (c == 0) break;
        }
    }
    return 0;
}

int goai_file_exists(const char *path) {
    if (!path || !*path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}
