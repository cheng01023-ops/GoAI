# GoAI - Makefile (the Xcode project builds the same sources)
CC      ?= clang
# 本机训练可以加 NATIVE=1 启用 -march=native（快 10~30%，但生成的二进制不能跨机器）
# 例： make clean && make NATIVE=1 all
ifeq ($(NATIVE),1)
  NATIVE_FLAGS := -march=native -mtune=native
endif
CFLAGS  ?= $(NATIVE_FLAGS) -O3 -ffast-math -pthread -std=c11 -Wall -Wextra -Wno-unused-parameter -Iinclude -MMD -MP -DGOAI_ENABLE_SOCKETS
TESTFLAGS ?= -O2 -pthread -std=c11 -Wall -Wextra -Wno-unused-parameter -Iinclude -MMD -MP
LDLIBS  ?= -lm
BUILD   := build

SRC      := src/board.c src/compat.c src/net.c src/mcts.c src/sockets.c src/remote.c src/gplay.c src/train.c src/apps.c
OBJ      := $(SRC:src/%.c=$(BUILD)/%.o) $(BUILD)/main.o
TESTSRC  := tests/test_main.c tests/test_board.c tests/test_rng.c tests/test_net.c tests/test_mcts.c tests/test_train.c
TESTOBJ  := $(TESTSRC:tests/%.c=$(BUILD)/test_%.o) $(BUILD)/board.o $(BUILD)/compat.o $(BUILD)/net.o $(BUILD)/mcts.o $(BUILD)/train.o

GOBOARD_OBJ := $(BUILD)/goboard.o $(BUILD)/board.o $(BUILD)/compat.o $(BUILD)/net.o $(BUILD)/mcts.o

.PHONY: all goboard goboard-text embed test bench clean run-train

all: $(BUILD)/GoAI

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/test_%.o: tests/%.c | $(BUILD)
	$(CC) $(TESTFLAGS) -c $< -o $@

$(BUILD)/GoAI: $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDLIBS)

# 本地人机对弈界面（ncurses）
goboard: $(BUILD)/GoBoard

$(BUILD)/GoBoard: $(GOBOARD_OBJ)
	$(CC) $(GOBOARD_OBJ) -o $@ $(LDLIBS) -lncurses

# 纯文本版人机对弈（不依赖 ncurses，适合在 VS Code 调试控制台里跑）
goboard-text: $(BUILD)/GoBoard-text

$(BUILD)/GoBoard-text: src/goboard.c src/board.c src/compat.c src/net.c src/mcts.c | $(BUILD)
	$(CC) $(CFLAGS) -DGOAI_NO_CURSES src/goboard.c src/board.c src/compat.c src/net.c src/mcts.c -o $@ $(LDLIBS)

# 把训练好的权重编译进对弈程序（默认取最新的训练输出）
#   make embed                        # 自动挑最新（runs9_s800/best.bin -> runs_step5/latest.bin ...）
#   make embed WEIGHTS=versions/goai9x9_v5_800sims.bin
WEIGHTS ?=
embed:
	python3 tools/embed_weights.py --latest $(if $(WEIGHTS),--from $(WEIGHTS),) --out include/weights_builtin.h
	@echo "重新编译生效: make goboard"

$(BUILD)/GoAITests: $(TESTOBJ)
	$(CC) $(TESTOBJ) -o $@ $(LDLIBS) -pthread

test: $(BUILD)/GoAITests
	./$(BUILD)/GoAITests

bench: $(BUILD)/GoAI
	./$(BUILD)/GoAI bench --size 9

run-train: $(BUILD)/GoAI
	./$(BUILD)/GoAI train --size 9 --iters 8 --games 40 --sims 120 --out runs

clean:
	rm -rf $(BUILD)

-include $(OBJ:.o=.d) $(TESTOBJ:.o=.d)
