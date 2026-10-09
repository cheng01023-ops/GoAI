# macOS / Linux：全屏棋盘版（需要 ncurses，macOS 自带）
CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -std=c11 -Wall -Wextra -Wno-unused-parameter
INC     := -Iinclude
SRC     := src/goboard.c src/board.c src/compat.c src/net.c src/mcts.c
BUILD   := build

.PHONY: all goboard goboard-text clean run test

all: goboard

# 全屏界面（ncurses + 方向键）
goboard: $(BUILD)/GoBoard

$(BUILD)/GoBoard: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(SRC) -o $@ -lm -lncurses

# 文本界面（无依赖，输入坐标落子）
goboard-text: $(BUILD)/GoBoard-text

$(BUILD)/GoBoard-text: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -DGOAI_NO_CURSES $(SRC) -o $@ -lm

$(BUILD):
	mkdir -p $(BUILD)

run: $(BUILD)/GoBoard
	./$(BUILD)/GoBoard --sims 200

test: $(BUILD)/GoBoard
	./$(BUILD)/GoBoard --selftest --sims 40

clean:
	rm -rf $(BUILD)
