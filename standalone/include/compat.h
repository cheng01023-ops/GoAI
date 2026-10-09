/* compat.h - 跨平台小工具（Windows / macOS / Linux 通用）
 *
 * 引擎原本使用了一些 POSIX 接口（sysconf / nanosleep / clock_gettime /
 * localtime_r / getpid / mkdir），这些在 Windows(MSVC) 上并不存在。
 * 这里统一封装，让同一份源码能在三平台上编译。
 */
#ifndef GOAI_COMPAT_H
#define GOAI_COMPAT_H

#include <stddef.h>

#ifdef _WIN32
#  ifdef GOAI_ENABLE_SOCKETS
#    include <winsock2.h>    /* 必须排在 windows.h 前面 */
#    include <ws2tcpip.h>
#  endif
#  include <direct.h>
#  include <process.h>
#  include <windows.h>
#  define GOAI_MKDIR_ONE(p) _mkdir(p)
#  define GOAI_PATHSEP '\\'
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#  define GOAI_MKDIR_ONE(p) mkdir((p), 0755)
#  define GOAI_PATHSEP '/'
#endif

/* 单调时钟，单位秒（用于计时） */
double goai_now(void);

/* 本地时间字符串，fmt 形如 "%Y-%m-%d %H:%M:%S" */
void   goai_localtime_str(char *buf, size_t n, const char *fmt);

/* 当前进程号 */
long   goai_pid(void);

/* 可用的逻辑 CPU 核数 */
int    goai_cpu_count(void);

/* 睡眠指定毫秒 */
void   goai_sleep_ms(int ms);

/* 递归创建目录（自动识别 '/' 与 '\\' 分隔符），已存在也算成功 */
int    goai_mkdir_p(const char *path);

/* ---------------- TCP 套接字 ----------------
 * 只有需要连 GPU 推理服务的程序（GoAI 的 gtrain 模式）才定义 GOAI_ENABLE_SOCKETS。
 * 对弈程序不需要网络，这样 Windows 上编译时就不会引用 Winsock、不需要 -lws2_32。
 */
#ifdef GOAI_ENABLE_SOCKETS
#ifdef GOAI_ENABLE_SOCKETS
#  ifdef _WIN32
typedef SOCKET goai_sock_t;
#    define GOAI_SOCK_INVALID INVALID_SOCKET
#  else
#    include <netdb.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
typedef int goai_sock_t;
#    define GOAI_SOCK_INVALID (-1)
#  endif
#endif /* GOAI_ENABLE_SOCKETS */

int  goai_net_init(void);          /* 进程启动时调一次 */
void goai_net_cleanup(void);
goai_sock_t goai_tcp_connect(const char *host, int port);   /* 失败返回 GOAI_SOCK_INVALID */
int  goai_sock_send_all(goai_sock_t s, const void *buf, size_t len);
int  goai_sock_recv_all(goai_sock_t s, void *buf, size_t len);
void goai_sock_close(goai_sock_t s);
#endif /* GOAI_ENABLE_SOCKETS */

/* 文件是否存在（用于跨平台的优雅停止标记） */
int    goai_file_exists(const char *path);

#endif /* GOAI_COMPAT_H */
