/* sockets.c - TCP 客户端（只被需要连 GPU 推理服务的程序链接）
 *
 * 单独成一个文件的原因：Windows 上必须链接 ws2_32 才能用 Winsock。
 * 对弈程序完全不需要网络，让它跟这个文件脱钩，就能做到"只要有 gcc 就能编译"。
 */
#define GOAI_ENABLE_SOCKETS 1
#include "compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* ---------------- TCP 套接字（按需编译，对弈程序用不到） ---------------- */

#ifdef _WIN32
int goai_net_init(void) {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0 ? 0 : -1;
}
void goai_net_cleanup(void) { WSACleanup(); }
void goai_sock_close(goai_sock_t s) { if (s != GOAI_SOCK_INVALID) closesocket(s); }
#else
int goai_net_init(void) { return 0; }
void goai_net_cleanup(void) {}
void goai_sock_close(goai_sock_t s) { if (s >= 0) close(s); }
#endif

#include <string.h>

goai_sock_t goai_tcp_connect(const char *host, int port) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints, *res = NULL, *p = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, portbuf, &hints, &res) != 0) return GOAI_SOCK_INVALID;
    goai_sock_t fd = GOAI_SOCK_INVALID;
    for (p = res; p; p = p->ai_next) {
        fd = (goai_sock_t)socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd == GOAI_SOCK_INVALID) continue;
        if (connect(fd, p->ai_addr, (int)p->ai_addrlen) == 0) break;
        goai_sock_close(fd);
        fd = GOAI_SOCK_INVALID;
    }
    freeaddrinfo(res);
    if (fd != GOAI_SOCK_INVALID) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));  /* 降低延迟 */
    }
    return fd;
}

int goai_sock_send_all(goai_sock_t s, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < len) {
        int n = (int)send(s, p + sent, (int)(len - sent), 0);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

int goai_sock_recv_all(goai_sock_t s, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t got = 0;
    while (got < len) {
        int n = (int)recv(s, p + got, (int)(len - got), 0);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    return 0;
}
