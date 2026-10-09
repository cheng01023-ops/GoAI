import csv, statistics
p = "$HOME/Desktop/新建文件夹/GoAI/runs_live/train_log.csv".replace("$HOME", __import__("os").path.expanduser("~"))
rows = list(csv.DictReader(open(p)))
print("总轮数:", len(rows), " 总对局:", sum(int(r["games"]) for r in rows))
def f(r, k):
    try: return float(r[k])
    except Exception: return None
print()
print("轮次   策略损失  价值损失  对随机胜率  对最佳胜率  每轮用时")
step = max(1, len(rows)//12)
for i in range(0, len(rows), step):
    r = rows[i]
    gw = f(r, "gate_winrate")
    gws = "  -  " if gw is None or gw < 0 else "%.0f%%" % (gw*100)
    print("%4s   %7.3f   %7.3f    %5.0f%%      %6s     %5ss" % (
        r["iteration"], f(r,"policy_loss"), f(r,"value_loss"), f(r,"winrate_vs_random")*100, gws, f(r,"elapsed_s")))
r = rows[-1]
print()
print("最新一轮:", r["iteration"], "策略损失 %.3f 价值损失 %.3f" % (f(r,"policy_loss"), f(r,"value_loss")))
first, last = rows[0], rows[-1]
print("策略损失: %.3f -> %.3f (降 %.0f%%)" % (f(first,"policy_loss"), f(last,"policy_loss"),
      (1-f(last,"policy_loss")/f(first,"policy_loss"))*100))
ws = [f(r,"winrate_vs_random") for r in rows if f(r,"winrate_vs_random") is not None]
print("对随机胜率: 最近 50 轮平均 %.0f%%（前 50 轮平均 %.0f%%）" % (
    statistics.mean(ws[-50:])*100, statistics.mean(ws[:50])*100))
gates = [f(r,"gate_winrate") for r in rows if f(r,"gate_winrate") is not None and f(r,"gate_winrate") >= 0]
if gates:
    print("晋级赛: 共 %d 次，平均 %.0f%%，高于 55%% 的 %d 次（说明在持续淘汰旧版本）" % (
        len(gates), statistics.mean(gates)*100, sum(1 for g in gates if g > 0.55)))
    print("  最近 8 次晋级赛:", " ".join("%.0f%%" % (g*100) for g in gates[-8:]))
els = [f(r,"elapsed_s") for r in rows[-100:] if f(r,"elapsed_s")]
print("每轮用时: 最近 100 轮平均 %.1fs" % statistics.mean(els))
