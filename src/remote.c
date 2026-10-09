/* remote.c - GPU 推理服务客户端 */
#include "remote.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REMOTE_MAGIC 0x49414F47u /* "GOAI" little-endian */

int remote_parse_addr(const char *s, char *host, size_t hostlen, int *port) {
    if (!s) return -1;
    const char *colon = strrchr(s, ':');
    if (!colon) { snprintf(host, hostlen, "%s", s); *port = 8899; return 0; }
    size_t n = (size_t)(colon - s);
    if (n >= hostlen) n = hostlen - 1;
    memcpy(host, s, n);
    host[n] = 0;
    *port = atoi(colon + 1);
    if (*port <= 0) *port = 8899;
    return 0;
}

int remote_connect(RemoteNet *r, const char *host, int port) {
    memset(r, 0, sizeof(*r));
    r->fd = GOAI_SOCK_INVALID;
    goai_net_init();
    for (int attempt = 0; attempt < 60; attempt++) {       /* 最多等 30 秒，方便服务端慢启动 */
        r->fd = goai_tcp_connect(host, port);
        if (r->fd != GOAI_SOCK_INVALID) break;
        goai_sleep_ms(500);
    }
    if (r->fd == GOAI_SOCK_INVALID) {
        fprintf(stderr, "remote: 连不上 %s:%d（GPU 服务端起了吗？）\n", host, port);
        return -1;
    }
    uint32_t hello[2] = { REMOTE_MAGIC, 1 };
    if (goai_sock_send_all(r->fd, hello, sizeof(hello)) != 0) return -1;
    int32_t hdr[6];
    if (goai_sock_recv_all(r->fd, hdr, sizeof(hdr)) != 0) {
        fprintf(stderr, "remote: 握手失败\n");
        return -1;
    }
    r->size = hdr[0]; r->planes = hdr[1]; r->channels = hdr[2];
    r->vhidden = hdr[3]; r->nn = hdr[4]; r->max_batch = hdr[5];
    return 0;
}

int remote_eval(RemoteNet *r, int count, const float *x, float *policy_out, float *value_out) {
    if (count <= 0) return 0;
    uint32_t n = (uint32_t)count;
    const size_t xlen = (size_t)count * r->planes * r->nn;
    if (goai_sock_send_all(r->fd, &n, sizeof(n)) != 0) return -1;
    if (goai_sock_send_all(r->fd, x, xlen * sizeof(float)) != 0) return -1;
    if (goai_sock_recv_all(r->fd, policy_out, (size_t)count * (r->nn + 1) * sizeof(float)) != 0) return -1;
    if (goai_sock_recv_all(r->fd, value_out, (size_t)count * sizeof(float)) != 0) return -1;
    return 0;
}

int remote_upload(RemoteNet *r, int count, const uint8_t *x, const float *pi, const float *z) {
    if (count <= 0) return 0;
    /* 最高位置 1 表示这是“上传训练样本”，不是评估请求 */
    uint32_t n = (uint32_t)count | 0x80000000u;
    if (goai_sock_send_all(r->fd, &n, sizeof(n)) != 0) return -1;
    if (goai_sock_send_all(r->fd, x, (size_t)count * r->planes * r->nn) != 0) return -1;
    if (goai_sock_send_all(r->fd, pi, (size_t)count * (r->nn + 1) * sizeof(float)) != 0) return -1;
    if (goai_sock_send_all(r->fd, z, (size_t)count * sizeof(float)) != 0) return -1;
    return 0;
}

void remote_close(RemoteNet *r) {
    goai_sock_close(r->fd);
    r->fd = GOAI_SOCK_INVALID;
    goai_net_cleanup();
}
