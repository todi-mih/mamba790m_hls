open_project -reset proj
set_top vadd_mul

# repo root = directory where vitis_hls was launched
set REPO [pwd]
set INC  "-I${REPO}/headers"

add_files kernel.cpp                 -cflags "${INC}"
add_files stages/rmsnorm_stage.cpp   -cflags "${INC}"
add_files stages/gemv_in_proj.cpp    -cflags "${INC}"
add_files stages/conv_stage.cpp      -cflags "${INC}"
add_files stages/gemv_x_proj.cpp     -cflags "${INC}"
add_files stages/gemv_dt_proj.cpp    -cflags "${INC}"
add_files stages/ssm_stage.cpp       -cflags "${INC}"
add_files stages/gating_stage.cpp    -cflags "${INC}"
add_files stages/gemv_out_proj.cpp   -cflags "${INC}"
add_files tb.cpp -tb                 -cflags "${INC}"

open_solution sol1
csim_design