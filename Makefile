CC ?= gcc
MPICC ?= mpicc
NVCC ?= nvcc

CFLAGS ?= -O3 -g -std=c11 -Wall -Wextra -fopenmp
LDFLAGS ?= -lm -fopenmp
CUDAFLAGS ?= -O3 -g
CUDA_HOME ?= /share/apps/hpc_sdk/Linux_x86_64/25.1/cuda
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64 -lcudart -lstdc++

MPIEXEC ?= mpirun
MPI_NP ?= 3
MPIEXEC_FLAGS ?= -np $(MPI_NP)

VTUNE ?= vtune
VTUNE_ANALYSIS ?= hotspots
VTUNE_MPI_ANALYSIS ?= hpc-performance
VTUNE_MPI_OPTIONS ?= -trace-mpi
VTUNE_DIR ?= vtune_results
ifeq ($(origin VTUNE_RUN), undefined)
VTUNE_RUN := $(shell date +%Y%m%d_%H%M%S)
endif
VTUNE_PART1_HOTSPOTS_RESULT ?= $(VTUNE_DIR)/part1_hotspots_$(VTUNE_RUN)
VTUNE_PART1_THREADING_RESULT ?= $(VTUNE_DIR)/part1_threading_$(VTUNE_RUN)
VTUNE_PART1_RESULT ?= $(VTUNE_DIR)/part1_$(VTUNE_ANALYSIS)_$(VTUNE_RUN)
VTUNE_PART2_RESULT ?= $(VTUNE_DIR)/part2_$(VTUNE_MPI_ANALYSIS)_$(VTUNE_RUN)
VTUNE_PART3_RESULT ?= $(VTUNE_DIR)/part3_$(VTUNE_MPI_ANALYSIS)_$(VTUNE_RUN)

BUILD_DIR := build
SRC_DIR := src

PART1_TARGET := $(BUILD_DIR)/wave_part1
PART1_SOURCES := $(SRC_DIR)/assignment_1.c

PART2_TARGET := $(BUILD_DIR)/wave_part2_mpi
PART2_SOURCES := $(SRC_DIR)/assignment_2.c

PART3_TARGET := $(BUILD_DIR)/wave_part3_cuda_mpi
PART3_HOST_OBJ := $(BUILD_DIR)/assignment_3.o
PART3_CUDA_OBJ := $(BUILD_DIR)/wave_color_cuda.o
PART3_HOST_SOURCE := $(SRC_DIR)/assignment_3.c
PART3_CUDA_SOURCE := $(SRC_DIR)/wave_color_cuda.cu

# GAMMA, C and AMPLITUDE below are student 352165's *assigned* values
# (parameters_part_1.pdf) -- do not change without checking the sheet.
# DT, DX, SIZE and STEPS are NOT assigned by any parameter sheet: they are
# free grid/time-discretization choices. These are the canonical values
# also used as assignment_3.c's compiled-in defaults, so both files agree.
GAMMA ?= 0.067
C ?= 0.59
DT ?= 0.02
DX ?= 0.02
SIZE ?= 400
STEPS ?= 400
AMPLITUDE ?= 56
SCALE ?= 5
THREADS ?= 4
OUTPUT ?= sim


.PHONY: all part1 part2 part3 run-smoke run-student run-part2 run-part3 \
	vtune-part1 vtune-part1-hotspots vtune-part1-threading \
	vtune-part1-analysis vtune-part2 vtune-part3 \
	vtune-report clean clean-vtune clean-part1-output clean-part2-output
all: part1 part2 part3

# --- output cleanup -----------------------------------------------------

clean-part1-output:
	rm -rf $(OUTPUT)

clean-part2-output:
	rm -rf sim1 sim2 sim3

clean-part3-output:
	rm -rf sim1_ppm sim2_ppm sim3_ppm

# --- build ---------------------------------------------------------------

part1: $(PART1_TARGET)

$(PART1_TARGET): $(PART1_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

part2: $(PART2_TARGET)

$(PART2_TARGET): $(PART2_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

part3: $(PART3_TARGET)

# Part 3 links a CUDA object (compiled with nvcc) and a host/MPI object
# (compiled with mpicc) into one executable.
$(PART3_CUDA_OBJ): $(PART3_CUDA_SOURCE) $(SRC_DIR)/wave_color_cuda.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

$(PART3_HOST_OBJ): $(PART3_HOST_SOURCE) $(SRC_DIR)/wave_color_cuda.h
	mkdir -p $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(PART3_TARGET): $(PART3_HOST_OBJ) $(PART3_CUDA_OBJ)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

# --- run -------------------------------------------------------------

# Small, fast sanity-check run with hardcoded toy parameters (not the
# assigned ones) -- just to confirm the binary runs end to end.
run-smoke: $(PART1_TARGET)
	$(PART1_TARGET) --gamma 0.05 --c 1.0 --dt 0.1 --dx 1.0 --size 64 --steps 10 --impulse-i 32 --impulse-j 32 --amplitude 1.0 --threads 2 --output sim

run-student: $(PART1_TARGET) clean-part1-output
	OMP_NUM_THREADS=$(THREADS) $(PART1_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --scale $(SCALE) --threads $(THREADS) --output $(OUTPUT)

run-part2: $(PART2_TARGET) clean-part2-output
	OMP_NUM_THREADS=$(THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) $(PART2_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output-prefix sim

run-part3: $(PART3_TARGET) clean-part3-output
	OMP_NUM_THREADS=$(THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) $(PART3_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output-prefix sim

# --- profiling (VTune; CPU/MPI side only -- see chat for Nsight on the
# CUDA side of part3) ------------------------------------------------

vtune-part1: $(PART1_TARGET)
	$(MAKE) --no-print-directory vtune-part1-hotspots VTUNE_RUN=$(VTUNE_RUN)
	$(MAKE) --no-print-directory vtune-part1-threading VTUNE_RUN=$(VTUNE_RUN)
	@echo "Part 1 VTune reports:"
	@echo "  $(VTUNE_PART1_HOTSPOTS_RESULT)_summary.txt"
	@echo "  $(VTUNE_PART1_THREADING_RESULT)_summary.txt"

vtune-part1-hotspots: $(PART1_TARGET)
	mkdir -p $(VTUNE_DIR)
	OMP_NUM_THREADS=$(THREADS) $(VTUNE) -collect hotspots -result-dir $(VTUNE_PART1_HOTSPOTS_RESULT) -- $(PART1_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output $(OUTPUT)
	$(VTUNE) -report summary -r $(VTUNE_PART1_HOTSPOTS_RESULT) > $(VTUNE_PART1_HOTSPOTS_RESULT)_summary.txt

vtune-part1-threading: $(PART1_TARGET)
	mkdir -p $(VTUNE_DIR)
	OMP_NUM_THREADS=$(THREADS) $(VTUNE) -collect threading -result-dir $(VTUNE_PART1_THREADING_RESULT) -- $(PART1_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output $(OUTPUT)
	$(VTUNE) -report summary -r $(VTUNE_PART1_THREADING_RESULT) > $(VTUNE_PART1_THREADING_RESULT)_summary.txt

vtune-part1-analysis: $(PART1_TARGET)
	mkdir -p $(VTUNE_DIR)
	OMP_NUM_THREADS=$(THREADS) $(VTUNE) -collect $(VTUNE_ANALYSIS) -result-dir $(VTUNE_PART1_RESULT) -- $(PART1_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output $(OUTPUT)
	$(VTUNE) -report summary -r $(VTUNE_PART1_RESULT) > $(VTUNE_PART1_RESULT)_summary.txt

vtune-part2: $(PART2_TARGET)
	mkdir -p $(VTUNE_DIR)
	OMP_NUM_THREADS=$(THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) $(VTUNE) -collect $(VTUNE_MPI_ANALYSIS) $(VTUNE_MPI_OPTIONS) -result-dir $(VTUNE_PART2_RESULT) -- $(PART2_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output-prefix sim
	@echo "VTune MPI result prefix: $(VTUNE_PART2_RESULT)"
	@echo "For a text report, run: make vtune-report RESULT=<actual_result_directory>"

vtune-part3: $(PART3_TARGET)
	mkdir -p $(VTUNE_DIR)
	OMP_NUM_THREADS=$(THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) $(VTUNE) -collect $(VTUNE_MPI_ANALYSIS) $(VTUNE_MPI_OPTIONS) -result-dir $(VTUNE_PART3_RESULT) -- $(PART3_TARGET) --gamma $(GAMMA) --c $(C) --dt $(DT) --dx $(DX) --size $(SIZE) --steps $(STEPS) --amplitude $(AMPLITUDE) --threads $(THREADS) --output-prefix sim
	@echo "VTune MPI/CUDA-host result prefix: $(VTUNE_PART3_RESULT)"
	@echo "For a text report, run: make vtune-report RESULT=<actual_result_directory>"

vtune-report:
	test -n "$(RESULT)"
	$(VTUNE) -report summary -r $(RESULT)

clean:
	rm -rf $(BUILD_DIR)

clean-vtune:
	rm -rf $(VTUNE_DIR)
