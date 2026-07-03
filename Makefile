# ─────────────────────────────────────────────
# Environment throttling
# ─────────────────────────────────────────────
export RDI_PARALLEL=2
export VITIS_VPL_SPAWN_MULTIPLE_JOBS=0

# ─────────────────────────────────────────────
# Platform / build config
# ─────────────────────────────────────────────
PLATFORM    = xilinx_u55n_gen3x4_xdma_1_202110_1
TARGET      = hw

XO          = kernel.xo
XCLBIN      = kernel.xclbin
HOST_EXE    = host

CFG         = connectivity.cfg
BUILD_CFG   = vitis_build.cfg

KERNEL      = vadd_mul

# ─────────────────────────────────────────────
# Sources / layout
# ─────────────────────────────────────────────
STAGE_DIR   = stages
HEADER_DIR  = headers

STAGE_SRCS  = $(STAGE_DIR)/rmsnorm_stage.cpp \
              $(STAGE_DIR)/gemv_in_proj.cpp \
              $(STAGE_DIR)/conv_stage.cpp \
              $(STAGE_DIR)/gemv_x_proj.cpp \
              $(STAGE_DIR)/gemv_dt_proj.cpp \
              $(STAGE_DIR)/ssm_stage.cpp \
              $(STAGE_DIR)/gating_stage.cpp \
              $(STAGE_DIR)/gemv_out_proj.cpp

KERNEL_SRCS = kernel.cpp $(STAGE_SRCS)
HEADERS     = $(wildcard $(HEADER_DIR)/*.h)

# ─────────────────────────────────────────────
# Toolchain
# ─────────────────────────────────────────────
VPP         = v++
CXX         = g++

XRT_INC     = /opt/xilinx/xrt/include
XRT_LIB     = /opt/xilinx/xrt/lib
VITIS_INC   = $(XILINX_VITIS)/include
VITIS_LIB   = $(XILINX_VITIS)/lib

CXXFLAGS    = -std=c++17 -O2 -Wall -I$(HEADER_DIR) -I$(XRT_INC) -I$(VITIS_INC)
LDFLAGS     = -L$(XRT_LIB) -L$(VITIS_LIB) -lxrt_coreutil -lpthread -ldl

VPPFLAGS    = --save-temps --target $(TARGET) --platform $(PLATFORM)
VPP_INC     = -I$(HEADER_DIR)

# ─────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────
.PHONY: all compile link host run clean cleanall

all: compile link host

# ── compile kernel (.xo) ─────────────────────
compile: $(XO)

$(XO): $(KERNEL_SRCS) $(HEADERS) $(CFG) $(BUILD_CFG)
	$(VPP) -c $(VPPFLAGS) $(VPP_INC) \
	--config $(CFG) \
	--config $(BUILD_CFG) \
	-k $(KERNEL) $(KERNEL_SRCS) \
	-o $@

# ── link (.xclbin) ───────────────────────────
link: $(XCLBIN)

$(XCLBIN): $(XO) $(CFG) $(BUILD_CFG)
	$(VPP) -l $(VPPFLAGS) \
	--config $(CFG) \
	--config $(BUILD_CFG) \
	$(XO) -o $@

# ── host ──────────────────────────────────────
host: $(HOST_EXE)

$(HOST_EXE): host.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) host.cpp -o $@ $(LDFLAGS)

# ── run ───────────────────────────────────────
run: all
	unset XCL_EMULATION_MODE && ./$(HOST_EXE) $(XCLBIN)

# ── clean ─────────────────────────────────────
clean:
	rm -f $(XO) $(XCLBIN) $(HOST_EXE)
	rm -rf _x *.log *.jou *.wdb *.wcfg *.summary *.html
	rm -f run_log.txt

cleanall: clean
	rm -rf .Xil .run .ipcache