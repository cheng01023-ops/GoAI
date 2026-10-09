# GoAI - Makefile (the Xcode project builds the same sources)
CC      ?= clang
CFLAGS  ?= -O3 -ffast-math -pthread -std=c11 -Wall -Wextra -Wno-unused-parameter -Iinclude -MMD -MP -DGOAI_ENABLE_SOCKETS
TESTFLAGS ?= -O2 -pthread -std=c11 -Wall -Wextra -Wno-unused-parameter -Iinclude -MMD -MP
LDLIBS  ?= -lm
BUILD   := build

SRC      := src/board.c src/compat.c src/net.c src/mcts.c src/sockets.c src/remote.c src/gplay.c src/train.c src/apps.c
OBJ      := $(SRC:src/%.c=$(BUILD)/%.o) $(BUILD)/main.o
TESTSRC  := tests/test_main.c tests/test_board.c tests/test_rng.c tests/test_net.c tests/test_mcts.c tests/test_train.c
TESTOBJ  := $(TESTSRC:tests/%.c=$(BUILD)/test_%.o) $(BUILD)/board.o $(BUILD)/compat.o $(BUILD)/net.o $(BUILD)/mcts.o $(BUILD)/train.o

GOBOARD_OBJ := $(BUILD)/goboard.o $(BUILD)/board.o $(BUILD)/compat.o $(BUILD)/net.o $(BUILD)/mcts.o

.PHONY: all goboard test bench clean run-train

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
