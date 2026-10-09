/* remote.h - 连接 GPU 推理/训练服务的客户端
 *
 * 协议（小端二进制，TCP）：
 *   握手：客户端发 "GOAI" + uint32 版本(1)
 *         服务端回 int32[6]: size, planes, channels, vhidden, nn, max_batch
 *   评估：客户端发 uint32 count + count*(planes*nn) float32 特征
 *         服务端回 count*(nn+1) float32 策略 + count float32 价值
 *   上传：客户端发 uint32 count + count*(planes*nn) uint8 特征 + count*(nn+1) float32 策略目标 + count float32 z
 *         （单向，不等回复）
 */
#ifndef GOAI_REMOTE_H
#define GOAI_REMOTE_H

#include <stdint.h>

#include "compat.h"

typedef struct {
    goai_sock_t fd;
    int size, planes, channels, vhidden, nn, max_batch;
} RemoteNet;

int  remote_connect(RemoteNet *r, const char *host, int port);
int  remote_eval(RemoteNet *r, int count, const float *x, float *policy_out, float *value_out);
int  remote_upload(RemoteNet *r, int count, const uint8_t *x, const float *pi, const float *z);
void remote_close(RemoteNet *r);

/* 便于 --remote host:port 解析 */
int  remote_parse_addr(const char *s, char *host, size_t hostlen, int *port);

#endif /* GOAI_REMOTE_H */
