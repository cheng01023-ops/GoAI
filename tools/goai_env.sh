#!/bin/bash
# GoAI 公共配置与函数（由各个 .command 入口 source）
# 可用环境变量覆盖：GOAI_OUT / GOAI_GAMES / GOAI_SIMS / GOAI_STEPS / GOAI_CHANNELS / GOAI_LR

GOAI_PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOAI_OUT="${GOAI_OUT:-$GOAI_PROJ/runs_live}"
GOAI_PIDFILE="$GOAI_OUT/train.pid"
GOAI_LOGFILE="$GOAI_OUT/train_console.log"
GOAI_CONF="$GOAI_PROJ/tools/goai.conf"

GOAI_GAMES="${GOAI_GAMES:-40}"
GOAI_SIMS="${GOAI_SIMS:-140}"
GOAI_STEPS="${GOAI_STEPS:-250}"
GOAI_CHANNELS="${GOAI_CHANNELS:-32}"
GOAI_LR="${GOAI_LR:-0.01}"
# 全力档专用：把单线程阶段（评估/晋级赛）压缩，让多核自对弈占更大比例
GOAI_EVALGAMES="${GOAI_EVALGAMES:-}"
GOAI_EVALSIMS="${GOAI_EVALSIMS:-}"
GOAI_GATE_EVERY="${GOAI_GATE_EVERY:-}"
GOAI_GATEGAMES="${GOAI_GATEGAMES:-}"

# 强度：auto / low / mid / high
goai_intensity() {
  local v="auto"
  [ -f "$GOAI_CONF" ] && v=$(cat "$GOAI_CONF" 2>/dev/null | tr -d ' \n')
  [ -z "$v" ] && v="auto"
  echo "$v"
}

goai_intensity_name() {
  case "$(goai_intensity)" in
    low)  echo "低（2 线程 · 几乎无感）";;
    mid)  echo "中（4 线程 · 均衡）";;
    high) echo "高（8 线程）";;
    max)  echo "全力榨干（满核 + 满优先级 + 防睡眠）";;
    *)    echo "自动（插电 8 / 电池 4）";;
  esac
}

# 满速模式：GOAI_FULLSPEED=1 时不加后台 QoS、用满 CPU（会跟你的正常使用抢性能）
goai_fullspeed() {
  [ "${GOAI_FULLSPEED:-0}" = "1" ] && return 0
  [ "$(goai_intensity)" = "max" ]   # 档位选了"全力榨干"也算满速
}

goai_threads() {
  local cpus threads mode
  cpus=$(sysctl -n hw.ncpu)
  mode=$(goai_intensity)
  case "$mode" in
    low) threads=2 ;;
    mid) threads=4 ;;
    high) threads=8 ;;
    max) threads=$cpus ;;
    *) if pmset -g batt 2>/dev/null | head -1 | grep -q "Battery Power"; then threads=$(( cpus / 2 )); else threads=$(( cpus - 2 )); fi ;;
  esac
  if goai_fullspeed; then threads=$cpus; fi
  [ "$threads" -lt 1 ] && threads=1
  [ "$threads" -gt 8 ] && { goai_fullspeed || threads=8; }
  echo "$threads"
}

goai_power_note() {
  if pmset -g batt 2>/dev/null | head -1 | grep -q "Battery Power"; then echo "电池供电"; else echo "外接电源"; fi
}

# 找真正的训练进程：先看训练器自报的 PID（status.txt），再看 PID 文件，最后按进程名兜底
goai_pid_is_trainer() {
  [ -n "$1" ] || return 1
  # 只看可执行文件本身，避免把“命令行里含 GoAI train 的 shell”误判成训练进程
  local comm
  comm=$(ps -o comm= -p "$1" 2>/dev/null)
  case "$comm" in
    */GoAI|GoAI) return 0 ;;
  esac
  return 1
}

goai_effective_pid() {
  local spid="" pid=""
  spid=$(goai_status_value pid)
  if goai_pid_is_trainer "$spid"; then echo "$spid"; return 0; fi
  [ -f "$GOAI_PIDFILE" ] && pid=$(cat "$GOAI_PIDFILE" 2>/dev/null)
  if goai_pid_is_trainer "$pid"; then echo "$pid"; return 0; fi
  local found; found=$(pgrep -x GoAI | head -1)   # 按进程名精确匹配
  if [ -n "$found" ]; then echo "$found"; return 0; fi
  return 1
}

goai_running() { goai_effective_pid >/dev/null; }

goai_start() {
  mkdir -p "$GOAI_OUT"
  # 注意：goai_threads 在 $() 子 shell 里跑，export 传不出来，这里显式设一次
  [ "$(goai_intensity)" = "max" ] && export GOAI_FULLSPEED=1
  if goai_running; then echo "训练已经在运行了（PID $(cat "$GOAI_PIDFILE")）"; return 1; fi
  echo "编译引擎…"
  ( cd "$GOAI_PROJ" && make -s all ) || { echo "❌ 编译失败，请查看上面的报错"; return 1; }
  local threads
  threads=$(goai_threads)
  # 评估/晋级赛是单线程的，会浪费其他核；满速档把它们压小，让多核自对弈占更大比例
  if goai_fullspeed; then
    GOAI_EVALGAMES="${GOAI_EVALGAMES:-8}"; GOAI_EVALSIMS="${GOAI_EVALSIMS:-20}"
    GOAI_GATE_EVERY="${GOAI_GATE_EVERY:-10}"; GOAI_GATEGAMES="${GOAI_GATEGAMES:-12}"
  else
    GOAI_EVALGAMES="${GOAI_EVALGAMES:-12}"; GOAI_EVALSIMS="${GOAI_EVALSIMS:-40}"
    GOAI_GATE_EVERY="${GOAI_GATE_EVERY:-3}"; GOAI_GATEGAMES="${GOAI_GATEGAMES:-20}"
  fi
  rm -f "$GOAI_PIDFILE"
  local NICEN=10
  goai_fullspeed && NICEN=0
  # 全力档额外用 caffeinate 阻止系统睡眠（否则合盖/闲置会让训练停下来）
  local CAF=""
  if goai_fullspeed && command -v caffeinate >/dev/null 2>&1; then CAF="caffeinate -s -i"; fi
  ( cd "$GOAI_PROJ" && nohup nice -n $NICEN $CAF ./build/GoAI train \
      --size 9 --channels "$GOAI_CHANNELS" --forever --resume "$GOAI_OUT/latest.bin" \
      --games "$GOAI_GAMES" --sims "$GOAI_SIMS" --steps "$GOAI_STEPS" \
      --batch 64 --lr "$GOAI_LR" --evalgames "$GOAI_EVALGAMES" --evalsims "$GOAI_EVALSIMS" \
      --threads "$threads" --sgf 2 --gate 1 --gategames "$GOAI_GATEGAMES" --eval-every "$GOAI_GATE_EVERY" --plot 1 \
      --out "$GOAI_OUT" < /dev/null >> "$GOAI_LOGFILE" 2>&1 & )
  sleep 1
  sleep 2
  local realpid
  realpid=$(goai_effective_pid 2>/dev/null)
  if [ -n "$realpid" ]; then
    echo "$realpid" > "$GOAI_PIDFILE"
    # 后台 QoS：你正常用电脑时，它只用能效核与空闲算力
    if ! goai_fullspeed; then
      command -v taskpolicy >/dev/null 2>&1 && taskpolicy -b -p "$realpid" >/dev/null 2>&1
    fi
  fi
  if goai_running; then
    echo "✅ 训练已启动（PID $(cat "$GOAI_PIDFILE")）"
    echo "   线程数 $threads · 强度：$(goai_intensity_name) · 供电：$(goai_power_note)"
    echo "   自对弈 $GOAI_GAMES 局/轮 · 每步 $GOAI_SIMS 次模拟 · 训练 $GOAI_STEPS 步/轮"
    echo "   评估 $GOAI_EVALGAMES 局/轮 · 每 $GOAI_GATE_EVERY 轮打一次晋级赛（$GOAI_GATEGAMES 局）"
    if goai_fullspeed; then echo "   模式：满速（不加后台 QoS，会跟正常使用抢 CPU）"; else echo "   模式：礼貌（nice 低优先级 + 后台 QoS，你忙时自动让出性能核）"; fi
    echo "   日志 $GOAI_LOGFILE"
  else
    echo "❌ 启动失败，日志末尾："; tail -5 "$GOAI_LOGFILE"
  fi
}

goai_stop() {
  if ! goai_running; then echo "当前没有正在运行的训练。"; rm -f "$GOAI_PIDFILE"; return 0; fi
  local pid i=0
  pid=$(goai_effective_pid)
  echo "正在请求停止：PID $pid"
  echo "会在当前这一轮结束后保存权重（通常几秒到一分钟）…"
  kill -TERM "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null && [ $i -lt 180 ]; do sleep 1; i=$((i+1)); done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null; echo "⚠️  等待超时，已强制结束（建议下次降低强度）"
  else
    echo "✅ 已安全停止；权重：$GOAI_OUT/latest.bin，最佳：$GOAI_OUT/best.bin"
  fi
  rm -f "$GOAI_PIDFILE"
}

goai_status_value() { grep "^$1=" "$GOAI_OUT/status.txt" 2>/dev/null | head -1 | cut -d= -f2-; }

goai_bar() {  # $1=done $2=total $3=width
  local done=${1:-0} total=${2:-1} w=${3:-24} fill i out=""
  [ "$total" -le 0 ] && total=1
  fill=$(( done * w / total ))
  i=0; while [ $i -lt $w ]; do
    if [ $i -lt $fill ]; then out="$out#"; else out="$out."; fi
    i=$((i+1))
  done
  echo "$out"
}

goai_min() { awk -v s="$1" 'BEGIN{printf "%.1f", s/60}'; }
goai_pct() { awk -v w="$1" 'BEGIN{printf "%.1f", w*100}'; }

goai_dashboard() {
  local f="$GOAI_OUT/status.txt" state iter game gi total ei et wr pid
  echo "╔══════════════════════════════════════════════════════════════╗"
  printf "║   GoAI 自我迭代训练监控            %s              ║\n" "$(date '+%H:%M:%S')"
  echo "╚══════════════════════════════════════════════════════════════╝"
  pid=$(goai_effective_pid 2>/dev/null)
  if [ -n "$pid" ]; then
    printf "  进程        运行中 · PID %s · %s 线程 · 强度 %s\n" \
      "$pid" "$(goai_status_value threads)" "$(goai_intensity_name)"
  else
    printf "  进程        未运行\n"
  fi
  if [ -f "$f" ]; then
    state=$(goai_status_value state)
    iter=$(goai_status_value iteration)
    game=$(goai_status_value game)
    gi=$(goai_status_value games_this_iter)
    total=$(goai_status_value total_games)
    et=$(goai_status_value elapsed_total)
    wr=$(goai_status_value winrate_vs_random)
    case "$state" in
      running)  state="训练中" ;;
      stopped)  state="已停止" ;;
      finished) state="已完成" ;;
    esac
    printf "  状态        %s\n" "$state"
    printf "  当前轮次    第 %s 轮   [%s] %s/%s 局\n" "$iter" "$(goai_bar "$game" "$gi" 24)" "$game" "$gi"
    printf "  累计对局    %s 局\n" "$total"
    [ -n "$et" ] && printf "  已运行      %s 分钟\n" "$(goai_min "$et")"
    [ -n "$wr" ] && printf "  最近评估    对随机胜率 %s%%\n" "$(goai_pct "$wr")"
    printf "  最后更新    %s\n" "$(goai_status_value updated)"
  else
    echo "  （还没有训练记录，选 1 开始训练）"
  fi
  echo
  echo "  训练曲线（每轮自动刷新）：$GOAI_OUT/training_curve.png"
  if [ -f "$GOAI_OUT/train_log.csv" ]; then
    echo
    echo "  轮次  局面数  策略损失  价值损失  对随机  用时"
    tail -n +2 "$GOAI_OUT/train_log.csv" | tail -6 | while IFS=, read -r it g pos pl vl pn wr2 gw el; do
      printf "  %-5s %-7s %-9s %-9s %-7s %ss\n" "$it" "$pos" "$pl" "$vl" "$(goai_pct "$wr2")%" "$el"
    done
  fi
}

goai_refresh_plot() {
  ( cd "$GOAI_PROJ" && python3 tools/plot.py "$GOAI_OUT/train_log.csv" "$GOAI_OUT/training_curve.png" >/dev/null 2>&1 )
  [ -f "$GOAI_OUT/training_curve.png" ] && open "$GOAI_OUT/training_curve.png"
}

goai_set_intensity() {
  echo "选择训练强度（决定用几个 CPU 核心做自对弈）："
  echo "  1) 低   （2 线程，几乎无感）"
  echo "  2) 中   （4 线程，均衡）"
  echo "  3) 高   （8 线程）"
  echo "  4) 自动 （插电 8 线程 / 电池 4 线程）"
  echo "  5) 全力榨干（满核 + 满优先级 + 防睡眠；插电整夜训练用这个）"
  printf "请选择 [1-5]: "
  read -r c
  case "$c" in
    1) echo low  > "$GOAI_CONF" ;;
    2) echo mid  > "$GOAI_CONF" ;;
    3) echo high > "$GOAI_CONF" ;;
    5) echo max  > "$GOAI_CONF" ;;
    *) echo auto > "$GOAI_CONF" ;;
  esac
  echo "已设置：$(goai_intensity_name)（下次启动训练时生效）"
}

goai_watch() {
  local key="" tick=0
  while true; do
    clear
    goai_dashboard
    echo
    if goai_running; then
      echo "  ▶ 正在训练…（每 2 秒刷新，按 q 返回菜单）"
    else
      echo "  ■ 训练未运行（按 q 返回菜单）"
    fi
    tick=$((tick+1))
    if [ $((tick % 8)) -eq 1 ]; then
      ( cd "$GOAI_PROJ" && python3 tools/plot.py "$GOAI_OUT/train_log.csv" "$GOAI_OUT/training_curve.png" >/dev/null 2>&1 )
    fi
    key=""
    read -t 2 -n 1 key 2>/dev/null || true
    [ "$key" = "q" ] && break
  done
}
