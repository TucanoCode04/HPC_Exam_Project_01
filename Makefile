CC := gcc
CFLAGS := -O3 -std=c11 -Wall -Wextra -fopenmp
LDFLAGS := -lm -fopenmp

BUILD_DIR := build
SRC_DIR := src
TARGET := $(BUILD_DIR)/wave_sim.exe
SOURCES := $(SRC_DIR)/main.c

GAMMA ?= 0.067
C ?= 0.59
DT ?= 0.1
DX ?= 1.0
SIZE ?= 512
STEPS ?= 300
AMPLITUDE ?= 56
THREADS ?= 4
OUTPUT ?= sim
FPS ?= 10
VIDEO ?= wave_352165.mp4

.PHONY: all clean run-smoke run-student install-ffmpeg video

all: $(TARGET)

$(TARGET): $(SOURCES)
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run-smoke: $(TARGET)
	$(TARGET) --gamma 0.05 --c 1.0 --dt 0.1 --dx 1.0 --size 64 --steps 10 --impulse-i 32 --impulse-j 32 --amplitude 1.0 --threads 2 --output sim

run-student: $(TARGET)
	$(TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output $(OUTPUT)

install-ffmpeg:
	powershell -ExecutionPolicy Bypass -File scripts/install_ffmpeg.ps1

video:
	powershell -ExecutionPolicy Bypass -File scripts/make_video.ps1 -Framerate $(FPS) -InputDir $(OUTPUT) -Output $(VIDEO) -Frames $(STEPS)

clean:
	@if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"
