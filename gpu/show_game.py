"""看一眼 AI 自己下的棋：开局、手数、结果、吃子"""
import glob, os, re, sys

d = os.path.expanduser("~/Desktop/新建文件夹/GoAI/runs_live/games")
files = sorted(glob.glob(os.path.join(d, "*.sgf")), key=os.path.getmtime)
if len(sys.argv) > 1:
    f = sys.argv[1]
else:
    f = files[-1]

s = open(f, encoding="utf-8", errors="replace").read()
size = int(re.search(r"SZ\[(\d+)\]", s).group(1))
moves = re.findall(r";([BW])\[([a-s]{0,2})\]", s)
cols = "ABCDEFGHJKLMNOPQRST"

def fmt(v):
    if not v:
        return "停着"
    x = ord(v[0]) - 97
    y = ord(v[1]) - 97
    return cols[x] + str(size - y)

print("棋谱:", os.path.basename(f), " %dx%d" % (size, size), " 共", len(moves), "手")
re_m = re.search(r"RE\[([^\]]*)\]", s)
print("结果:", re_m.group(1) if re_m else "?")
print()
print("前 30 手:")
line = []
for i, (c, v) in enumerate(moves[:30]):
    line.append("%d.%s%s" % (i + 1, c, fmt(v)))
    if len(line) == 10:
        print("  " + " ".join(line))
        line = []
if line:
    print("  " + " ".join(line))

# 质量指标：与前一手距离（近身接触 = 在打仗，而不是乱丢）
def xy(v):
    return (ord(v[0]) - 97, ord(v[1]) - 97)
dist = []
for i in range(1, len(moves)):
    if moves[i][1] and moves[i - 1][1]:
        a, b = xy(moves[i][1]), xy(moves[i - 1][1])
        dist.append(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
if dist:
    near = sum(1 for x in dist if x <= 2)
    print()
    print("落子与前一手距离 <=2 的比例: %.0f%%  (高 = 在贴身战斗/应手，低 = 乱下)" % (100.0 * near / len(dist)))
    print("平均距离: %.1f 格" % (sum(dist) / len(dist)))

passes = sum(1 for c, v in moves if not v)
print("停着次数:", passes)
print()
print("最近 8 局:")
for g in files[-8:]:
    t = open(g, encoding="utf-8", errors="replace").read()
    n = len(re.findall(r";[BW]\[", t))
    r = re.search(r"RE\[([^\]]*)\]", t)
    print("  %-22s %3d 手  %s" % (os.path.basename(g), n, r.group(1) if r else "?"))
