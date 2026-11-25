import{r as R,j as c,c as dt}from"./react-CI6SGl1L.js";import{S as pt,a as _t,D as Ae,E as le,H as gt,t as I,b as ht,c as Ge,l as vt,h as xt,p as yt,s as wt,d as kt,k as bt,e as St,f as jt,o as Rt}from"./editor-CbRXp0Hd.js";(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))n(o);new MutationObserver(o=>{for(const m of o)if(m.type==="childList")for(const _ of m.addedNodes)_.tagName==="LINK"&&_.rel==="modulepreload"&&n(_)}).observe(document,{childList:!0,subtree:!0});function r(o){const m={};return o.integrity&&(m.integrity=o.integrity),o.referrerPolicy&&(m.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?m.credentials="include":o.crossOrigin==="anonymous"?m.credentials="omit":m.credentials="same-origin",m}function n(o){if(o.ep)return;o.ep=!0;const m=r(o);fetch(o.href,m)}})();const Nt=`!memref_gmem_f32 = !cute.memref<f32, gmem, align<16>, "(4096):(1)">
#loc = loc("src/sample/sample.py":19:0)
#loc1 = loc("src/sample/sample.py":9:0)
#loc11 = loc("<module>"(#loc))
#loc12 = loc("<module>"(#loc1))
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0(%arg0: !memref_gmem_f32 loc("<module>"(#loc1)), %arg1: !memref_gmem_f32 loc("<module>"(#loc1)), %arg2: !memref_gmem_f32 loc("<module>"(#loc1))) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, cutlass_launch_facts = {cluster_launch = false, cooperative_launch = false, exact_block_dim = array<i64: 128, 1, 1>, exact_grid_dim = array<i64: 32, 1, 1>, schema_version = 1 : i64}, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, smem.hint_smem_base_uniform, smem.partition_num = 2 : i32} {
      %c128_i32 = arith.constant 128 : i32 loc(#loc11)
      %0 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc13)
      %1 = nvvm.read.ptx.sreg.ctaid.x : i32 loc(#loc14)
      %2 = arith.muli %1, %c128_i32 : i32 loc(#loc15)
      %3 = arith.addi %2, %0 : i32 loc(#loc15)
      %coord = cute.make_coord(%3) : (i32) -> !cute.coord<"?"> loc(#loc16)
      %4 = cute.memref.load(%arg0, %coord) : (!memref_gmem_f32, !cute.coord<"?">) -> f32 loc(#loc16)
      %5 = cute.memref.load(%arg1, %coord) : (!memref_gmem_f32, !cute.coord<"?">) -> f32 loc(#loc17)
      %6 = arith.addf %4, %5 : f32 loc(#loc18)
      cute.memref.store(%arg2, %coord, %6) : (!memref_gmem_f32, !cute.coord<"?">, f32) -> () loc(#loc19)
      return loc(#loc11)
    } loc(#loc12)
  } loc(#loc11)
  func.func @cutlass_vector_add_FakeTensorFloat3240961_FakeTensorFloat3240961_FakeTensorFloat3240961(%arg0: !memref_gmem_f32 loc("<module>"(#loc)), %arg1: !memref_gmem_f32 loc("<module>"(#loc)), %arg2: !memref_gmem_f32 loc("<module>"(#loc))) -> i32 attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32 loc(#loc9)
    %c32_i32 = arith.constant 32 : i32 loc(#loc11)
    %c1_i32 = arith.constant 1 : i32 loc(#loc11)
    %c128_i32 = arith.constant 128 : i32 loc(#loc11)
    %c0_i64 = arith.constant 0 : i64 loc(#loc11)
    %c166912_i64 = arith.constant 166912 : i64 loc(#loc11)
    %0 = cute.kernel_smem_size @kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0 : i64 loc(#loc12)
    %1 = arith.cmpi sgt, %0, %c166912_i64 : i64 loc(#loc20)
    scf.if %1 {
      cute.print("\\0AError: kernel '@kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0' launch shared memory exceeds current GPU arch sm_80 allowed. Allocated: {} bytes. Max: 166912 bytes.\\0A\\0A", %0) : i64 loc(#loc11)
    } loc(#loc12)
    %2 = cuda.cast %c0_i64 : i64 -> !cuda.stream loc(#loc12)
    %3 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c128_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %0, gridDim = (%c32_i32, %c1_i32, %c1_i32), stream = %2) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17> loc(#loc12)
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%3] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32 loc(#loc12)
    cuda.launch_cfg.cooperative[%3] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32 loc(#loc12)
    %4 = cuda.launch_ex @kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0<%3> (%arg0, %arg1, %arg2) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!memref_gmem_f32, !memref_gmem_f32, !memref_gmem_f32) -> !cuda.result loc(#loc12)
    %5 = cuda.cast %4 : !cuda.result -> i32 loc(#loc12)
    cuda.return_if_error %5 : i32 loc(#loc12)
    return %c0_i32 : i32 loc(#loc11)
  } loc(#loc11)
} loc(#loc11)
#loc2 = loc("src/sample/sample.py":11:19)
#loc3 = loc("src/sample/sample.py":12:18)
#loc4 = loc("src/sample/sample.py":13:8)
#loc5 = loc("src/sample/sample.py":14:10)
#loc6 = loc("src/sample/sample.py":15:10)
#loc7 = loc("src/sample/sample.py":16:13)
#loc8 = loc("src/sample/sample.py":16:4)
#loc9 = loc(unknown)
#loc10 = loc("src/sample/sample.py":21:4)
#loc13 = loc("thread, _, _ = cute.arch.thread_idx()"(#loc2))
#loc14 = loc("block, _, _ = cute.arch.block_idx()"(#loc3))
#loc15 = loc("i = block * THREADS_PER_BLOCK + thread"(#loc4))
#loc16 = loc("lhs = a[i]"(#loc5))
#loc17 = loc("rhs = b[i]"(#loc6))
#loc18 = loc("out[i] = lhs + rhs"(#loc7))
#loc19 = loc("out[i] = lhs + rhs"(#loc8))
#loc20 = loc("vector_add_kernel(a, b, out).launch("(#loc10))

`,Et=`//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-38657139
// Cuda compilation tools, release 13.4, V13.4.59
// Based on NVVM 23.0.0
//

.version 9.4
.target sm_80
.address_size 64

	// .globl	kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0

.visible .entry kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0(
	.param .align 8 .b8 kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_0[8],
	.param .align 8 .b8 kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_1[8],
	.param .align 8 .b8 kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_2[8]
)
.reqntid 128, 1, 1
{
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<8>;
	.loc	1 9 0

	ld.param.b64 	%rd1, [kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_0];
	ld.param.b64 	%rd2, [kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_1];
	ld.param.b64 	%rd3, [kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0_param_2];
	.loc	1 11 19
	mov.u32 	%r1, %tid.x;
	.loc	1 12 18
	mov.u32 	%r2, %ctaid.x;
	.loc	1 13 8
	shl.b32 	%r3, %r2, 7;
	or.b32 	%r4, %r3, %r1;
	.loc	1 14 10
	mul.wide.s32 	%rd4, %r4, 4;
	add.s64 	%rd5, %rd1, %rd4;
	ld.global.b32 	%r5, [%rd5];
	.loc	1 15 10
	add.s64 	%rd6, %rd2, %rd4;
	ld.global.b32 	%r6, [%rd6];
	.loc	1 16 13
	add.f32 	%r7, %r5, %r6;
	.loc	1 16 4
	add.s64 	%rd7, %rd3, %rd4;
	st.global.b32 	[%rd7], %r7;
	.loc	1 19 0
	ret;

}
	.file	1 "src/sample/sample.py"
`,Mt=`import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor

THREADS_PER_BLOCK = 128
VECTOR_LENGTH = 4096


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * THREADS_PER_BLOCK + thread
    lhs = a[i]
    rhs = b[i]
    out[i] = lhs + rhs


@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    vector_add_kernel(a, b, out).launch(
        grid=(VECTOR_LENGTH // THREADS_PER_BLOCK, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )


def compile_for_explorer():
    a, b, out = (
        make_fake_tensor(cutlass.Float32, (VECTOR_LENGTH,), stride=(1,), assumed_align=16)
        for _ in range(3)
    )
    return cute.compile(vector_add, a, b, out, no_jit_engine=True)
`,At=`!memref_gmem_f32 = !cute.memref<f32, gmem, align<16>, "(4096):(1)">
#loc = loc("src/sample/sample.py":19:0)
#loc1 = loc("src/sample/sample.py":9:0)
#loc11 = loc("<module>"(#loc))
#loc12 = loc("<module>"(#loc1))
module attributes {gpu.container_module} {
  gpu.module @kernels {
    cuda.kernel @kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0(%arg0: !memref_gmem_f32 loc("<module>"(#loc1)), %arg1: !memref_gmem_f32 loc("<module>"(#loc1)), %arg2: !memref_gmem_f32 loc("<module>"(#loc1))) attributes {cu_attrs = {max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}, cute.kernel, cutlass_launch_facts = {cluster_launch = false, cooperative_launch = false, exact_block_dim = array<i64: 128, 1, 1>, exact_grid_dim = array<i64: 32, 1, 1>, schema_version = 1 : i64}, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, smem.hint_smem_base_uniform, smem.partition_num = 2 : i32} {
      %iter = cute.get_iter(%arg0) : !memref_gmem_f32 loc(#loc13)
      %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f32 loc(#loc13)
      %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f32 loc(#loc13)
      %iter_2 = cute.get_iter(%arg0) : !memref_gmem_f32 loc(#loc13)
      %iter_3 = cute.get_iter(%arg1) : !memref_gmem_f32 loc(#loc13)
      %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f32 loc(#loc13)
      %lay = cute.get_layout(%arg0) : !memref_gmem_f32 loc(#loc13)
      %lay_5 = cute.get_layout(%arg1) : !memref_gmem_f32 loc(#loc13)
      %lay_6 = cute.get_layout(%arg2) : !memref_gmem_f32 loc(#loc13)
      %0 = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc14)
      %1 = nvvm.read.ptx.sreg.tid.y : i32 loc(#loc14)
      %2 = nvvm.read.ptx.sreg.tid.z : i32 loc(#loc14)
      %3 = nvvm.read.ptx.sreg.ctaid.x : i32 loc(#loc15)
      %4 = nvvm.read.ptx.sreg.ctaid.y : i32 loc(#loc15)
      %5 = nvvm.read.ptx.sreg.ctaid.z : i32 loc(#loc15)
      %c128_i32 = arith.constant 128 : i32 loc(#loc11)
      %6 = arith.muli %3, %c128_i32 : i32 loc(#loc16)
      %7 = arith.addi %6, %0 : i32 loc(#loc16)
      %coord = cute.make_coord(%7) : (i32) -> !cute.coord<"?"> loc(#loc17)
      %8 = cute.memref.load(%arg0, %coord) : (!memref_gmem_f32, !cute.coord<"?">) -> f32 loc(#loc17)
      %coord_7 = cute.make_coord(%7) : (i32) -> !cute.coord<"?"> loc(#loc18)
      %9 = cute.memref.load(%arg1, %coord_7) : (!memref_gmem_f32, !cute.coord<"?">) -> f32 loc(#loc18)
      %10 = arith.addf %8, %9 : f32 loc(#loc19)
      %coord_8 = cute.make_coord(%7) : (i32) -> !cute.coord<"?"> loc(#loc20)
      cute.memref.store(%arg2, %coord_8, %10) : (!memref_gmem_f32, !cute.coord<"?">, f32) -> () loc(#loc20)
      return loc(#loc11)
    } loc(#loc12)
  } loc(#loc11)
  func.func @cutlass_vector_add_FakeTensorFloat3240961_FakeTensorFloat3240961_FakeTensorFloat3240961(%arg0: !memref_gmem_f32 loc("<module>"(#loc)), %arg1: !memref_gmem_f32 loc("<module>"(#loc)), %arg2: !memref_gmem_f32 loc("<module>"(#loc))) -> i32 attributes {llvm.emit_c_interface} {
    %iter = cute.get_iter(%arg0) : !memref_gmem_f32 loc(#loc21)
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f32 loc(#loc21)
    %iter_1 = cute.get_iter(%arg2) : !memref_gmem_f32 loc(#loc21)
    %iter_2 = cute.get_iter(%arg0) : !memref_gmem_f32 loc(#loc21)
    %iter_3 = cute.get_iter(%arg1) : !memref_gmem_f32 loc(#loc21)
    %iter_4 = cute.get_iter(%arg2) : !memref_gmem_f32 loc(#loc21)
    %lay = cute.get_layout(%arg0) : !memref_gmem_f32 loc(#loc13)
    %lay_5 = cute.get_layout(%arg1) : !memref_gmem_f32 loc(#loc13)
    %lay_6 = cute.get_layout(%arg2) : !memref_gmem_f32 loc(#loc13)
    %0 = cute.kernel_smem_size @kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0 : i64 loc(#loc12)
    %c166912_i64 = arith.constant 166912 : i64 loc(#loc11)
    %1 = arith.cmpi sgt, %0, %c166912_i64 : i64 loc(#loc13)
    scf.if %1 {
      cute.print("\\0AError: kernel '@kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0' launch shared memory exceeds current GPU arch sm_80 allowed. Allocated: {} bytes. Max: 166912 bytes.\\0A\\0A", %0) : i64 loc(#loc11)
    } loc(#loc12)
    %c0_i64 = arith.constant 0 : i64 loc(#loc11)
    %2 = cuda.cast %c0_i64 : i64 -> !cuda.stream loc(#loc12)
    %c128_i32 = arith.constant 128 : i32 loc(#loc11)
    %c1_i32 = arith.constant 1 : i32 loc(#loc11)
    %c32_i32 = arith.constant 32 : i32 loc(#loc11)
    %3 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c128_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %0, gridDim = (%c32_i32, %c1_i32, %c1_i32), stream = %2) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17> loc(#loc12)
    %c0_i32 = arith.constant 0 : i32 loc(#loc11)
    cuda.launch_cfg.programmatic_stream_serialization_allowed[%3] %c0_i32 : !cuda.launch_cfg<max_attrs = 17>, i32 loc(#loc12)
    %c0_i32_7 = arith.constant 0 : i32 loc(#loc12)
    cuda.launch_cfg.cooperative[%3] %c0_i32_7 : !cuda.launch_cfg<max_attrs = 17>, i32 loc(#loc12)
    %4 = cuda.launch_ex @kernels::@kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0<%3> (%arg0, %arg1, %arg2) {assume_kernel_attr = #cuda.assume_kernel_attr<true>} : !cuda.launch_cfg<max_attrs = 17>, (!memref_gmem_f32, !memref_gmem_f32, !memref_gmem_f32) -> !cuda.result loc(#loc12)
    %5 = cuda.cast %4 : !cuda.result -> i32 loc(#loc12)
    cuda.return_if_error %5 : i32 loc(#loc12)
    %c0_i32_8 = arith.constant 0 : i32 loc(#loc11)
    return %c0_i32_8 : i32 loc(#loc11)
  } loc(#loc11)
} loc(#loc11)
#loc2 = loc("src/sample/sample.py":21:4)
#loc3 = loc("src/sample/sample.py":11:19)
#loc4 = loc("src/sample/sample.py":12:18)
#loc5 = loc("src/sample/sample.py":13:8)
#loc6 = loc("src/sample/sample.py":14:10)
#loc7 = loc("src/sample/sample.py":15:10)
#loc8 = loc("src/sample/sample.py":16:13)
#loc9 = loc("src/sample/sample.py":16:4)
#loc10 = loc("src/sample/sample.py":32:11)
#loc13 = loc("vector_add_kernel(a, b, out).launch("(#loc2))
#loc14 = loc("thread, _, _ = cute.arch.thread_idx()"(#loc3))
#loc15 = loc("block, _, _ = cute.arch.block_idx()"(#loc4))
#loc16 = loc("i = block * THREADS_PER_BLOCK + thread"(#loc5))
#loc17 = loc("lhs = a[i]"(#loc6))
#loc18 = loc("rhs = b[i]"(#loc7))
#loc19 = loc("out[i] = lhs + rhs"(#loc8))
#loc20 = loc("out[i] = lhs + rhs"(#loc9))
#loc21 = loc("return cute.compile(vector_add, a, b, out, no_jit_engine=True)"(#loc10))

`,Tt=`	.target	sm_80

	.elftype	@"ET_EXEC"


//--------------------- .text.kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0 --------------------------
	.section	.text.kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0,"ax",@progbits
	.sectioninfo	@"SHI_REGISTERS=12"
	.align	128
        .global         kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0
        .type           kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0,@function
        .size           kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0,(.L_x_1 - kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0)
        .other          kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0,@"STO_CUDA_ENTRY STV_DEFAULT"
kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0:
.text.kernel_cutlass_vector_add_kernel_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_tensorptrf32gmemalign16o40961_0:
	//## File "src/sample/sample.py", line 9
        /*0000*/                   MOV R1, c[0x0][0x28] ;
	//## File "src/sample/sample.py", line 11
        /*0010*/                   S2R R6, SR_TID.X ;
	//## File "src/sample/sample.py", line 14
        /*0020*/                   HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07 ;
        /*0030*/                   ULDC.64 UR4, c[0x0][0x118] ;
	//## File "src/sample/sample.py", line 12
        /*0040*/                   S2R R3, SR_CTAID.X ;
	//## File "src/sample/sample.py", line 13
        /*0050*/                   LEA R6, R3, R6, 0x7 ;
	//## File "src/sample/sample.py", line 14
        /*0060*/                   IMAD.WIDE R2, R6, R7, c[0x0][0x160] ;
	//## File "src/sample/sample.py", line 15
        /*0070*/                   IMAD.WIDE R4, R6.reuse, R7.reuse, c[0x0][0x168] ;
	//## File "src/sample/sample.py", line 14
        /*0080*/                   LDG.E R2, [R2.64] ;
	//## File "src/sample/sample.py", line 15
        /*0090*/                   LDG.E R5, [R4.64] ;
	//## File "src/sample/sample.py", line 16
        /*00a0*/                   IMAD.WIDE R6, R6, R7, c[0x0][0x170] ;
        /*00b0*/                   FADD R9, R2, R5 ;
        /*00c0*/                   STG.E [R6.64], R9 ;
	//## File "src/sample/sample.py", line 19
        /*00d0*/                   EXIT ;
.L_x_0:
        /*00e0*/                   BRA \`(.L_x_0);
        /*00f0*/                   NOP;
        /*0100*/                   NOP;
        /*0110*/                   NOP;
        /*0120*/                   NOP;
        /*0130*/                   NOP;
        /*0140*/                   NOP;
        /*0150*/                   NOP;
        /*0160*/                   NOP;
        /*0170*/                   NOP;
.L_x_1:
`,rt=pt.define(),$t=_t.define({create:()=>Ae.none,update(e,t){for(const r of t.effects)if(r.is(rt))return Ae.set(r.value.filter(n=>n.line>0&&n.line<=t.state.doc.lines).map(n=>Ae.line({class:`correlation-${n.quality}`}).range(t.state.doc.line(n.line).from)),!0);return e.map(t.changes)},provide:e=>le.decorations.from(e)}),Ct=gt.define([{tag:[I.comment,I.lineComment,I.blockComment],color:"var(--syntax-comment)"},{tag:[I.keyword,I.controlKeyword,I.operatorKeyword],color:"var(--syntax-keyword)"},{tag:[I.number,I.bool,I.null],color:"var(--syntax-number)"},{tag:[I.typeName,I.className,I.atom],color:"var(--syntax-type)"},{tag:[I.string,I.special(I.string)],color:"var(--syntax-string)"},{tag:[I.variableName,I.labelName],color:"var(--syntax-var)"},{tag:I.definition(I.variableName),color:"var(--syntax-keyword)"}]),Dt=ht.define({name:"GPU compiler text",startState:()=>({comment:!1}),token(e,t){return t.comment?(e.skipTo("*/")?(e.match("*/"),t.comment=!1):e.skipToEnd(),"comment"):e.eatSpace()?null:e.match("//")?(e.skipToEnd(),"comment"):e.match("/*")?(t.comment=!0,"comment"):e.match(/^"(?:[^"\\]|\\.)*"/)?"string":e.match(/^[%@!#][\w.$-]+/)?"variableName":e.match(/^(?:0x[\da-fA-F]+|[-+]?\d+(?:\.\d+)?)(?![\w])/)?"number":e.match(/^(?:[.][\w.$-]+|(?:arith|cute|cute_nvgpu|func|gpu|llvm|memref|nvvm|scf|vector)\.[\w.]+|[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*\b|(?:ld|st|mov|add|sub|mul|mad|fma|cvt|bar|fence|ret|bra|setp)(?:\.[\w]+)*\b)/)?"keyword":e.match(/^(?:f16|f32|f64|i1|i32|i64|index|vector|tensor|memref)\b/)?"typeName":(e.match(/^[\w.$-]+/)||e.next(),null)}});function Lt({stage:e,text:t,marks:r,targetLine:n,onLine:o,searchHandle:m}){const _=R.useRef(null),p=R.useRef(null),s=R.useRef(o);return s.current=o,R.useLayoutEffect(()=>{const f=new le({parent:_.current,state:Ge.create({doc:t,extensions:[vt({domEventHandlers:{click(d,i){return s.current(d.state.doc.lineAt(i.from).number),!1}}}),le.editable.of(!1),Ge.readOnly.of(!0),le.contentAttributes.of({tabindex:"0","aria-label":`${e} code`}),xt(),e==="python"?yt():Dt,wt(Ct),$t,kt({top:!0}),bt.of(St),le.domEventHandlers({click(d,i){const l=d.target instanceof Element?d.target.closest(".cm-line"):null;return l&&s.current(i.state.doc.lineAt(i.posAtDOM(l)).number),!1},keydown(d,i){return d.key==="Enter"&&d.target instanceof Element&&d.target.closest(".cm-content")?(s.current(i.state.doc.lineAt(i.state.selection.main.head).number),!0):!1}})]})});return p.current=f,m.current=()=>{f.dom.querySelector(".cm-search")?jt(f):Rt(f)},()=>{m.current=null,p.current=null,f.destroy()}},[e,t,m]),R.useLayoutEffect(()=>{const f=p.current;if(!f)return;const d=[rt.of(r)];n&&n<=f.state.doc.lines&&d.push(le.scrollIntoView(f.state.doc.line(n).from,{y:"nearest",x:"nearest"})),f.dispatch({effects:d})},[r,n,t]),c.jsx("div",{className:"code-host",ref:_})}const qe=["python","raw","clean","ptx","sass","report"],Te=({stage:e,line:t})=>`${e}:${t}`,Xe=(e,t)=>`${t}:${e}`,Ee=e=>e.replace(/\r?\n$/,"").split(/\r?\n/);function Oe(e,t,r=0){if(!(!e||!Number.isSafeInteger(t)||t<=0))return Number.isSafeInteger(r)&&r>0?{file:e,line:t,column:r}:{file:e,line:t}}const It=({file:e,line:t,column:r})=>`${r??0}:${t}:${e}`,ze=e=>[...new Map(e.map(t=>[It(t),t])).values()],nt=/^#[\w$-]+$/,ue=/^\d+$/,de=String.raw`"(?:[^"\\\r\n]|\\.)*"`;function Ne(e,t){const r=e.slice(1,-1);return t==="mlir"?r.replace(/(?:\\[0-9a-f]{2})+|\\["\\nrt]/gi,n=>{if(n.length>2){const o=n.match(/[0-9a-f]{2}/gi)??[];return new TextDecoder().decode(Uint8Array.from(o,m=>parseInt(m,16)))}return{n:`
`,r:"\r",t:"	"}[n[1]]??n[1]}):t==="ptx"?r.replace(/\\(["\\])/g,"$1"):r}function Ft(e){const t=new RegExp(String.raw`${de}|//.*|->|[#%@!]?[\w$.-]+|[^\s]`,"g");return Ee(e).flatMap((r,n)=>{const o=r.search(/\S/);return[...r.matchAll(t)].filter(m=>!m[0].startsWith("//")).map(m=>({value:m[0],line:n+1,indent:o}))})}function Pt(e,t){const r=()=>{var p;return((p=e[m])==null?void 0:p.value)??""},n=p=>r()!==p?!1:(m++,!0),o=[];let m=t+2;function _(p){if(p>128)return!1;if(n("unknown")||n("?"))return!0;if(n("loc"))return n("(")&&_(p+1)&&n(")");if(n("callsite"))return n("(")&&_(p+1)&&n("at")&&_(p+1)&&n(")");if(n("fused")){if(n("<")){let l=1;for(;r()&&l>0;){const k=e[m++].value;k==="<"&&l++,k===">"&&l--}if(l)return!1}if(!n("["))return!1;if(n("]"))return!0;do if(!_(p+1))return!1;while(n(","));return n("]")}const s=r();if(nt.test(s))return o.push(s),m++,!0;if(!s.startsWith('"')||s.length<2)return!1;if(m++,!n(":"))return!n("(")||_(p+1)&&n(")");if(!ue.test(r()))return!1;const f=Number(e[m++].value);let d=0;if(n(":")){if(!ue.test(r()))return!1;if(d=Number(e[m++].value),n("to")){if(ue.test(r())&&m++,!n(":")||!ue.test(r()))return!1;m++}}const i=Oe(Ne(s,"mlir"),f,d);return i&&o.push(i),!0}return _(0)&&n(")")?{end:m,parts:o}:void 0}function Ot(e,t){var d,i,l,k,b,x,w,h,S;const r=new Map,n=new Map,o=Ft(e);let m=0,_;const p=[];let s;for(let v=0;v<o.length;){const j=o[v],C=(d=o[v+1])==null?void 0:d.value;if(j.line>m){m=j.line,s=nt.test(j.value)&&C==="="?j.value:void 0;let y=v;for(;((i=o[y])==null?void 0:i.line)===m&&((l=o[y])!=null&&l.value.startsWith("%"))&&(y++,((k=o[y])==null?void 0:k.value)===":"&&ue.test(((b=o[y+1])==null?void 0:b.value)??"")&&(y+=2),((x=o[y])==null?void 0:x.value)===",");)y++;const D=y>v&&((w=o[y])==null?void 0:w.line)===m&&((h=o[y])==null?void 0:h.value)==="=",E=/^[\w][\w.-]*$/.test(j.value)||j.value.startsWith('"'),U=["loc","{","}",")","]",":","->","else"].includes(j.value);s||j.value.startsWith("^")?_=void 0:!U&&(D||E||!_||j.indent<=_.indent)&&(_=j)}if(j.value==="{"?p.push(_):j.value==="}"&&(_=p.pop()),j.value==="loc"&&C==="("){const y=Pt(o,v);if(y){for(const E of y.parts)typeof E!="string"&&t(E.file);const D=o[y.end-1].line;if(s)r.set(s,y.parts);else if(_&&((S=o[v-1])==null?void 0:S.value)!=="="){const E=_.line;n.set(E,[...n.get(E)??[],...y.parts])}v=y.end,m=D;continue}}v++}const f=new Map;for(const[v,j]of n){const C=[],y=new Set,D=[...j].reverse();for(;D.length;){const E=D.pop();if(typeof E!="string")C.push(E);else if(!y.has(E)){y.add(E);const U=r.get(E)??[];for(let P=U.length-1;P>=0;P--)D.push(U[P])}}C.length&&f.set(v,ze(C))}return f}function zt(e,t){const r=e.replace(new RegExp(String.raw`${de}|//[^\r\n]*|/\*[\s\S]*?\*/`,"g"),x=>x.startsWith("/")?x.replace(/[^\r\n]/g," "):x),n=Ee(r),o=new RegExp(String.raw`^\s*\.file\s+(\d+)\s+(${de})`),m=new Map;for(const x of n){const w=o.exec(x);if(!w)continue;const h=Number(w[1]),S=Ne(w[2],"ptx");t(S),m.set(h,!m.has(h)||m.get(h)===S?S:void 0)}const _=(x,w,h)=>Oe(m.get(Number(x))??"",Number(w),Number(h??0)),p=new Map;let s=[],f=!1,d=0,i=!1;const l=/^(?:\.[\w]+\s+)*\.(?:entry|func)\b/;let k=!n.some(x=>l.test(x.trimStart()));const b=new RegExp(de,"g");for(const[x,w]of n.entries()){let h=w.trim();if(l.test(h)&&(s=[],f=!0,d=0,i=!1),f){const j=h.indexOf("{");if(j<0){h.endsWith(";")&&(f=!1);continue}d=1,f=!1,h=h.slice(j+1).trimStart()}if(!d&&/^\.section\b/.test(h)&&(k=!1),!d&&!k)continue;if(/^\.loc\b/.test(h)){const j=/^\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?(?=$|[\s,;])/.exec(h);s=[];const C=j&&_(j[1],j[2],j[3]);if(C){s.push(C);for(const y of h.matchAll(/\binlined_at\s+(\d+)\s+(\d+)\s+(\d+)(?=$|[\s,;])/g)){const D=_(y[1],y[2],y[3]);D&&s.push(D)}s=ze(s)}continue}const S=h.replace(b,'""'),v=S.replace(/^(?:[{}]\s*|[\w$.-]+:(?!:)\s*)+/,"");!i&&/^(?:@!?[%\w$]+\s+)?[a-z][a-z\d._]*(?:::[a-z\d._]+)*(?=\s|;|$)/i.test(v)?(s.length&&p.set(x+1,s),i=!S.includes(";")):i&&S.includes(";")&&(i=!1);for(const j of S.matchAll(/[{}]/g))d+=j[0]==="{"?1:-1;d<=0&&!k&&(d=0,s=[],i=!1)}return p}const ot=".nv_debug_ptx_txt";function Bt(e,t){const r=new Map,n=String.raw`(${de}),\s*line\s+(\d+)(?:,\s*column\s+(\d+))?`,o=new RegExp(String.raw`^\s*//##\s*File\s+${n}`),m=new RegExp(String.raw`\binlined at\s+${n}`,"g"),_=/^\s*\/\*\s*[\da-f]+\s*\*\/\s*\{?\s*(?:@!?[\w]+\s+)?[A-Z][A-Z\d._]*(?=\s|;)/;let p=!0,s=[],f,d;for(const[i,l]of Ee(e).entries()){const k=/^\s*\.section\s+([^\s,]+)/.exec(l);if(k){p=/^\.text(?:\.|$)/.test(k[1]),s=[],f=void 0,d=void 0;continue}if(!p)continue;const b=o.exec(l);if(b){const w=S=>{const v=Ne(S[1],"sass");return t(v),Oe(v,Number(S[2]),Number(S[3]??0))},h=w(b);if(Ne(b[1],"sass")===ot)d=h;else{f??(f=[]),h&&f.push(h);for(const S of l.slice(b[0].length).matchAll(m)){const v=w(S);v&&f.push(v)}}continue}if(!_.test(l))continue;f&&(s=ze(f),f=void 0);const x=d?[...s,d]:s;x.length&&r.set(i+1,x)}return r}function Ut({stage:e,name:t,text:r},n){if(e==="raw"||e==="clean")return Ot(r,n);if(e==="ptx")return zt(r,n);if(e==="sass")return Bt(r,n);const o=new Map;return e==="python"&&t&&r&&Ee(r).forEach((m,_)=>o.set(_+1,[{file:t,line:_+1}])),o}const fe=e=>e.split(/[/\\]/).at(-1)??"";function Ht(e,t=[]){const r=new Map,n=new Map,o=new Map,m=new Map,_=new Set(e.filter(f=>f.stage!=="python").map(f=>fe(f.name)));_.add(ot);const p=(f,d,i)=>{const l=f.get(d)??new Set;l.add(i),f.set(d,l)},s=f=>{const d=fe(f);f!==d&&p(m,d,f)};for(const f of t)p(o,fe(f),f);for(const f of e){f.stage==="python"?p(o,fe(f.name),f.name):s(f.name);for(const[d,i]of Ut(f,s)){const l={stage:f.stage,line:d},k=Te(l);r.set(k,i);for(const b of i){const x=Xe(b.file,b.line),w=n.get(x)??new Map;w.set(k,l),n.set(x,w)}}}return{spots(f){return(r.get(Te(f))??[]).map(d=>({...d}))},related(f){const d=Te(f),i=new Map,l=(b,x,w,h)=>{for(const[S,v]of n.get(Xe(b,x))??[])h&&v.stage==="python"!=(h==="python")||S!==d&&(w==="metadata"||!i.has(S))&&i.set(S,{ref:{...v},quality:w})};for(const{file:b,line:x}of r.get(d)??[]){const w=fe(b),h=m.get(w),S=o.get(w),v=b===w,j=((h==null?void 0:h.size)??0)>1,C=((S==null?void 0:S.size)??0)>1;if((!v||!j&&!C)&&l(b,x,"metadata"),!w||_.has(w)||j||C)continue;const y=h==null?void 0:h.values().next().value,D=S==null?void 0:S.values().next().value;if(f.stage==="python")v||l(w,x,"approximate","compiler"),y&&y!==b&&l(y,x,"approximate","compiler");else{D&&D!==b&&l(D,x,"approximate","python");const E=v?y:w;E&&l(E,x,"approximate","compiler")}}const k=[...i.values()].sort(({ref:b},{ref:x})=>qe.indexOf(b.stage)-qe.indexOf(x.stage)||b.line-x.line);return[{ref:{...f},quality:"metadata"},...k]}}}var Z=Uint8Array,ce=Uint16Array,Zt=Int32Array,at=new Z([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),st=new Z([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),Kt=new Z([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),lt=function(e,t){for(var r=new ce(31),n=0;n<31;++n)r[n]=t+=1<<e[n-1];for(var o=new Zt(r[30]),n=1;n<30;++n)for(var m=r[n];m<r[n+1];++m)o[m]=m-r[n]<<5|n;return{b:r,r:o}},ct=lt(at,2),it=ct.b,Wt=ct.r;it[28]=258,Wt[258]=28;var Vt=lt(st,0),Gt=Vt.b,Ie=new ce(32768);for(var N=0;N<32768;++N){var J=(N&43690)>>1|(N&21845)<<1;J=(J&52428)>>2|(J&13107)<<2,J=(J&61680)>>4|(J&3855)<<4,Ie[N]=((J&65280)>>8|(J&255)<<8)>>1}var pe=(function(e,t,r){for(var n=e.length,o=0,m=new ce(t);o<n;++o)e[o]&&++m[e[o]-1];var _=new ce(t);for(o=1;o<t;++o)_[o]=_[o-1]+m[o-1]<<1;var p;if(r){p=new ce(1<<t);var s=15-t;for(o=0;o<n;++o)if(e[o])for(var f=o<<4|e[o],d=t-e[o],i=_[e[o]-1]++<<d,l=i|(1<<d)-1;i<=l;++i)p[Ie[i]>>s]=f}else for(p=new ce(n),o=0;o<n;++o)e[o]&&(p[o]=Ie[_[e[o]-1]++]>>15-e[o]);return p}),ge=new Z(288);for(var N=0;N<144;++N)ge[N]=8;for(var N=144;N<256;++N)ge[N]=9;for(var N=256;N<280;++N)ge[N]=7;for(var N=280;N<288;++N)ge[N]=8;var mt=new Z(32);for(var N=0;N<32;++N)mt[N]=5;var qt=pe(ge,9,1),Xt=pe(mt,5,1),$e=function(e){for(var t=e[0],r=1;r<e.length;++r)e[r]>t&&(t=e[r]);return t},V=function(e,t,r){var n=t/8|0;return(e[n]|e[n+1]<<8)>>(t&7)&r},Ce=function(e,t){var r=t/8|0;return(e[r]|e[r+1]<<8|e[r+2]<<16)>>(t&7)},Yt=function(e){return(e+7)/8|0},Be=function(e,t,r){return(t==null||t<0)&&(t=0),(r==null||r>e.length)&&(r=e.length),new Z(e.subarray(t,r))},Jt=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],H=function(e,t,r){var n=new Error(t||Jt[e]);if(n.code=e,Error.captureStackTrace&&Error.captureStackTrace(n,H),!r)throw n;return n},Qt=function(e,t,r,n){var o=e.length,m=n?n.length:0;if(!o||t.f&&!t.l)return r||new Z(0);var _=!r,p=_||t.i!=2,s=t.i;_&&(r=new Z(o*3));var f=function(we){var ke=r.length;if(we>ke){var be=new Z(Math.max(ke*2,we));be.set(r),r=be}},d=t.f||0,i=t.p||0,l=t.b||0,k=t.l,b=t.d,x=t.m,w=t.n,h=o*8;do{if(!k){d=V(e,i,1);var S=V(e,i+1,3);if(i+=3,S)if(S==1)k=qt,b=Xt,x=9,w=5;else if(S==2){var y=V(e,i,31)+257,D=V(e,i+10,15)+4,E=y+V(e,i+5,31)+1;i+=14;for(var U=new Z(E),P=new Z(19),z=0;z<D;++z)P[Kt[z]]=V(e,i+z*3,7);i+=D*3;for(var Q=$e(P),he=(1<<Q)-1,ve=pe(P,Q,1),z=0;z<E;){var te=ve[V(e,i,he)];i+=te&15;var v=te>>4;if(v<16)U[z++]=v;else{var B=0,O=0;for(v==16?(O=3+V(e,i,3),i+=2,B=U[z-1]):v==17?(O=3+V(e,i,7),i+=3):v==18&&(O=11+V(e,i,127),i+=7);O--;)U[z++]=B}}var ie=U.subarray(0,y),A=U.subarray(y);x=$e(ie),w=$e(A),k=pe(ie,x,1),b=pe(A,w,1)}else H(1);else{var v=Yt(i)+4,j=e[v-4]|e[v-3]<<8,C=v+j;if(C>o){s&&H(0);break}p&&f(l+j),r.set(e.subarray(v,C),l),t.b=l+=j,t.p=i=C*8,t.f=d;continue}if(i>h){s&&H(0);break}}p&&f(l+131072);for(var re=(1<<x)-1,ne=(1<<w)-1,Y=i;;Y=i){var B=k[Ce(e,i)&re],L=B>>4;if(i+=B&15,i>h){s&&H(0);break}if(B||H(2),L<256)r[l++]=L;else if(L==256){Y=i,k=null;break}else{var oe=L-254;if(L>264){var z=L-257,K=at[z];oe=V(e,i,(1<<K)-1)+it[z],i+=K}var ae=b[Ce(e,i)&ne],se=ae>>4;ae||H(3),i+=ae&15;var A=Gt[se];if(se>3){var K=st[se];A+=Ce(e,i)&(1<<K)-1,i+=K}if(i>h){s&&H(0);break}p&&f(l+131072);var xe=l+oe;if(l<A){var me=m-A,ye=Math.min(A,xe);for(me+l<0&&H(3);l<ye;++l)r[l]=n[me+l]}for(;l<xe;++l)r[l]=r[l-A]}}t.l=k,t.p=Y,t.b=l,t.f=d,k&&(d=1,t.m=x,t.d=b,t.n=w)}while(!d);return l!=r.length&&_?Be(r,0,l):r.subarray(0,l)},er=new Z(0),q=function(e,t){return e[t]|e[t+1]<<8},W=function(e,t){return(e[t]|e[t+1]<<8|e[t+2]<<16|e[t+3]<<24)>>>0},De=function(e,t){return W(e,t)+W(e,t+4)*4294967296};function tr(e,t){return Qt(e,{i:2},t&&t.out,t&&t.dictionary)}var Fe=typeof TextDecoder<"u"&&new TextDecoder,rr=0;try{Fe.decode(er,{stream:!0}),rr=1}catch{}var nr=function(e){for(var t="",r=0;;){var n=e[r++],o=(n>127)+(n>223)+(n>239);if(r+o>e.length)return{s:t,r:Be(e,r-1)};o?o==3?(n=((n&15)<<18|(e[r++]&63)<<12|(e[r++]&63)<<6|e[r++]&63)-65536,t+=String.fromCharCode(55296|n>>10,56320|n&1023)):o&1?t+=String.fromCharCode((n&31)<<6|e[r++]&63):t+=String.fromCharCode((n&15)<<12|(e[r++]&63)<<6|e[r++]&63):t+=String.fromCharCode(n)}};function or(e,t){if(t){for(var r="",n=0;n<e.length;n+=16384)r+=String.fromCharCode.apply(null,e.subarray(n,n+16384));return r}else{if(Fe)return Fe.decode(e);var o=nr(e),m=o.s,r=o.r;return r.length&&H(8),m}}var ar=function(e,t){return t+30+q(e,t+26)+q(e,t+28)},sr=function(e,t,r){var n=q(e,t+28),o=q(e,t+30),m=or(e.subarray(t+46,t+46+n),!(q(e,t+8)&2048)),_=t+46+n,p=lr(e,_,o,r,W(e,t+20),W(e,t+24),W(e,t+42)),s=p[0],f=p[1],d=p[2];return[q(e,t+10),s,f,m,_+o+q(e,t+32),d]},lr=function(e,t,r,n,o,m,_){var p=o==4294967295,s=m==4294967295,f=_==4294967295,d=t+r,i=p+s+f;if(n&&i){for(;t+4<d;t+=4+q(e,t+2))if(q(e,t)==1)return[p?De(e,t+4+8*s):o,s?De(e,t+4):m,f?De(e,t+4+8*(s+p)):_,1];n<2&&H(13)}return[o,m,_,0]};function cr(e,t){for(var r={},n=e.length-22;W(e,n)!=101010256;--n)(!n||e.length-n>65558)&&H(13);var o=q(e,n+8);if(!o)return{};var m=W(e,n+16),_=W(e,n-20)==117853008;if(_){var p=W(e,n-12);_=W(e,p)==101075792,_&&(o=W(e,p+32),m=W(e,p+48))}for(var s=t&&t.filter,f=0;f<o;++f){var d=sr(e,m,_),i=d[0],l=d[1],k=d[2],b=d[3],x=d[4],w=d[5],h=ar(e,w);m=x,(!s||s({name:b,size:l,originalSize:k,compression:i}))&&(i?i==8?r[b]=tr(e.subarray(h,h+l),{out:new Z(k)}):H(14,"unknown compression type "+i):r[b]=Be(e,h,h+l))}return r}const Ye=8*1024*1024,Je=32*1024*1024,Le=256;function Re(e){const t=e.toLowerCase(),r=t.split(/[\\/]/).at(-1)??t;if(r.endsWith(".py"))return"python";if(r.endsWith(".mlir"))return/(?:^|[._-])(?:raw|debug)\.mlir$/.test(r)?"raw":"clean";if(r.endsWith(".ptx"))return"ptx";if(/\.(?:sass|disasm|s)$/.test(r))return"sass";if(/\.(?:txt|log)$/.test(r))return"report"}async function ir(e){if(e.length>Le)throw new Error("File count limit exceeded.");let t=0,r=0;const n=[],o=new TextDecoder("utf-8",{fatal:!0});function m(p,s){if(t+=s,s>Ye||t>Je)throw new Error(`Text size limit exceeded: ${p} (8 MiB per file, 32 MiB total).`)}function _(p,s,f){let d;try{d=o.decode(f)}catch{throw new Error(`${s} is not UTF-8 text.`)}if(d.includes("\0"))throw new Error(`${s} is not a plain-text artifact.`);if(d.trim()?n.push({stage:p,name:s,text:d}):r++,n.length>Le)throw new Error("File count limit exceeded.")}for(const p of e)if(p.name.toLowerCase().endsWith(".zip")){if(p.size>Je||p.size<22)throw new Error(`Invalid ZIP or ZIP size limit exceeded: ${p.name}.`);const s=new Uint8Array(await p.arrayBuffer()),f=new Set;let d=0,i;try{i=cr(s,{filter:({name:l,originalSize:k})=>{if(++d>Le)throw new Error("ZIP entry count limit exceeded.");if(!Re(l)||/(?:^|\/)__MACOSX(?:\/|$)/.test(l)||/(?:^|\/)\._/.test(l))return r++,!1;if(f.has(l))throw new Error("ZIP contains duplicate artifact paths.");return f.add(l),m(l,k),!0}})}catch(l){throw l instanceof Error&&/limit exceeded|duplicate artifact/.test(l.message)?l:new Error(`Could not read ZIP: ${p.name}.`)}for(const[l,k]of Object.entries(i)){if(k.length>Ye)throw new Error(`ZIP entry size limit exceeded: ${l}.`);_(Re(l),l,k)}}else{const s=Re(p.name);if(!s){r++;continue}m(p.name,p.size),_(s,p.name,new Uint8Array(await p.arrayBuffer()))}if(!n.length)throw new Error("No supported nonempty text artifacts found. Upload Python, MLIR, PTX, SASS, a resource report, or a ZIP.");return{artifacts:n,skipped:r}}const mr=Object.assign({"./sample/sample.mlir":Nt,"./sample/sample.ptx":Et,"./sample/sample.py":Mt,"./sample/sample.raw.mlir":At,"./sample/sample.sass":Tt}),Se=Object.entries(mr).flatMap(([e,t])=>{const r=e.split("/").at(-1),n=Re(r);return n?[{stage:n,name:r,text:t}]:[]}),fr={python:"CuTeDSL",raw:"MLIR raw",clean:"MLIR clean",ptx:"PTX",sass:"SASS",report:"Report"},ur=[{id:"python",title:"CuTeDSL",hint:"Python source (.py)"},{id:"mlir",title:"MLIR",hint:"Raw and clean IR (.mlir)"},{id:"ptx",title:"PTX",hint:"Virtual ISA (.ptx)"},{id:"sass",title:"SASS",hint:"Disassembly (.sass, .disasm)"}],Pe=e=>e.split(/[\\/]/).at(-1)??e,je=e=>`${e.stage}:${e.line}`,Qe={mma:{python:/\b(?:dot|gemm)\s*\(/,mlir:/^\s*(?:%[\w.$-]+\s*=\s*)?(?:lir\.dot|cute\.gemm|cute_nvgpu\.mma)(?=\s|\()/,ptx:/^(?:tcgen05\.mma|mma\.sync|wgmma\.mma_async)\./,sass:/^(?:UTCHMMA|HMMA|IMMA|BMMA|DMMA)(?=\s|\.)/},tmaLoad:{python:/\btma_load\s*\(/,mlir:/^\s*(?:%[\w.$-]+\s*=\s*)?lir\.tma_load(?=\s|\()/,ptx:/^cp\.async\.bulk\.tensor\.[\w.:]*\.shared::(?:cta|cluster)\.global(?=\s|\.)/,sass:/^UTMALDG(?=\s|\.)/}};function et(e,t){if(e==="report")return;const r=e==="clean"||e==="raw"?"mlir":e,n=e==="python"?t.split("#",1)[0]:e==="ptx"?t.replace(/^\s*(?:[\w$.-]+:(?!:)\s*)?(?:@!?%[\w$]+\s+)?/,""):e==="sass"?t.replace(/^\s*\/\*\s*[\da-f]+\s*\*\/\s*(?:@!?\w+\s+)?/,""):t;return Object.keys(Qe).find(o=>Qe[o][r].test(n))}function _e({name:e}){const t={upload:"M12 16V4m-4 4 4-4 4 4M5 15v5h14v-5",search:"M15.5 15.5 21 21M17 10.5a6.5 6.5 0 1 1-13 0 6.5 6.5 0 0 1 13 0Z",expand:"M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5",restore:"M4 9h5V4m6 0v5h5M4 15h5v5m6 0v-5h5",theme:"M12 4a8 8 0 1 0 8 8 6.5 6.5 0 0 1-8-8Z",code:"m8 6-5 6 5 6m8-12 5 6-5 6m-5 3 2-6"};return c.jsx("svg",{width:"17",height:"17",viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:"1.6",strokeLinecap:"round",strokeLinejoin:"round","aria-hidden":"true",children:c.jsx("path",{d:t[e]})})}function dr({pane:e,stage:t,artifact:r,choices:n,pick:o,expanded:m,expand:_,controls:p,marks:s,target:f,select:d}){const i=R.useRef(null);return c.jsxs("section",{className:"pane","data-pane":e.id,"data-stage":t,"aria-label":`${e.title} code pane`,children:[c.jsxs("header",{className:"pane-header",children:[c.jsx("span",{className:`stage-dot ${e.id}`,"aria-hidden":"true"}),c.jsx("h2",{children:e.title}),n.length>1?c.jsx("select",{className:"artifact-name","aria-label":`${e.title} artifact`,value:r==null?void 0:r.name,onChange:l=>o(l.target.value),children:n.map(l=>c.jsx("option",{value:l.name,children:l.name},l.name))}):c.jsx("span",{className:"artifact-name",title:r==null?void 0:r.name,children:r?Pe(r.name):"Not loaded"}),c.jsxs("div",{className:"pane-actions",children:[p,c.jsx("button",{className:"icon","aria-label":`Search ${e.title}`,title:`Search ${e.title} (Ctrl/⌘+F)`,disabled:!r,onClick:()=>{var l;return(l=i.current)==null?void 0:l.call(i)},children:c.jsx(_e,{name:"search"})}),c.jsx("button",{className:"icon","aria-label":m?"Restore panes":`Expand ${e.title}`,title:m?"Restore panes":`Expand ${e.title}`,onClick:_,children:c.jsx(_e,{name:m?"restore":"expand"})})]})]}),r?c.jsx(Lt,{stage:t,text:r.text,marks:s,targetLine:f,onLine:l=>d({stage:t,line:l}),searchHandle:i}):c.jsxs("div",{className:"empty",children:[c.jsxs("span",{children:["No ",e.title," artifact"]}),c.jsxs("small",{children:["Load ",e.hint]})]})]})}const pr=()=>{try{return localStorage.getItem("cute-explorer-theme")==="light"?"light":"dark"}catch{return"dark"}};function _r(){var Ue;const[e,t]=R.useState(Se),[r,n]=R.useState(Se.length>0),[o,m]=R.useState({}),[_,p]=R.useState("clean"),[s,f]=R.useState(null),[d,i]=R.useState(null),[l,k]=R.useState(!0),[b,x]=R.useState(pr),[w,h]=R.useState(0),[S,v]=R.useState(""),[j,C]=R.useState(""),[y,D]=R.useState(!1),[E,U]=R.useState({col:50,row:50}),[P,z]=R.useState(null),Q=R.useRef(null),he=R.useRef(null),ve=R.useRef(null),te=R.useRef(!1),B=R.useMemo(()=>gr(e),[e]),O=R.useMemo(()=>{const a={};for(const u of Object.keys(B)){const g=B[u];a[u]=g.find(T=>T.name===o[u])??g[0]}return a},[B,o]),ie=O[_]?_:O.raw?"raw":"clean",A=R.useMemo(()=>Object.values(O),[O]),re=R.useMemo(()=>(B.python??[]).map(a=>a.name),[B]),ne=R.useMemo(()=>Ht(A,re),[A,re]),Y=R.useMemo(()=>new Map(A.map(a=>[a.stage,a.text.split(/\r?\n/)])),[A]),L=(P==null?void 0:P.input)===A?P.result:void 0,oe=(P==null?void 0:P.input)===A?P.error:void 0;R.useEffect(()=>{if(!A.length)return;const a=()=>z({input:A,error:"Analysis could not finish. The code panes and source links are still available."});let u;try{u=new Worker(new URL(""+new URL("analysis.worker-B5S9Z7gW.js",import.meta.url).href,import.meta.url),{type:"module"}),u.onmessage=g=>{g.data?z({input:A,result:g.data}):a(),u==null||u.terminate()},u.onerror=()=>{a(),u==null||u.terminate()},u.postMessage({artifacts:A,pythonNames:re})}catch{a(),u==null||u.terminate()}return()=>u==null?void 0:u.terminate()},[A,re]),R.useEffect(()=>{var a;s!=null&&s.refs.some(u=>u.stage==="report")&&((a=ve.current)==null||a.focus())},[s]);const K=R.useMemo(()=>{var Ze,Ke,We,Ve;const a={},u={},g={},T=new Map,M=F=>{var G;return((G=Y.get(F.stage))==null?void 0:G[F.line-1])??""},X=s==null?void 0:s.refs[0],$=!(s!=null&&s.finding)&&X&&et(X.stage,M(X));for(const F of(s==null?void 0:s.refs)??[])for(const G of ne.related(F)){const Me=je(G.ref);(!T.has(Me)||T.get(Me).quality==="approximate")&&T.set(Me,G)}for(const F of(s==null?void 0:s.refs)??[])T.set(je(F),{ref:F,quality:"selected"}),u[Ze=F.stage]??(u[Ze]=F.line);let ee=0,He=0;for(const{ref:F,quality:G}of T.values())(a[Ke=F.stage]??(a[Ke]=[])).push({line:F.line,quality:G}),u[We=F.stage]??(u[We]=F.line),$&&G!=="selected"&&et(F.stage,M(F))===$&&(g[Ve=F.stage]??(g[Ve]=F.line)),G==="metadata"&&ee++,G==="approximate"&&He++;return{result:a,destination:{...u,...g},metadata:ee,approximate:He}},[ne,s,Y]),ae=s==null?void 0:s.refs[0],se=(s==null?void 0:s.refs.filter(a=>a.stage==="report"))??[],xe=(s==null?void 0:s.message)??(ae?ne.spots(ae).length?K.metadata+K.approximate===0?"Source location found; no related lines in the loaded artifacts.":`${K.metadata} metadata-linked · ${K.approximate} approximate`:"No source metadata for the selected line.":"Click a line or line number to trace its source location.");function me(a){t(a?Se:[]),n(a),m({}),p("clean"),f(null),v(""),C("")}async function ye(a){if(!(!a.length||te.current)){te.current=!0,D(!0),v(""),C("");try{const{artifacts:u,skipped:g}=await ir(a),T=new Map((r?[]:e).map($=>[`${$.stage}:${$.name}`,$]));for(const $ of u)T.set(`${$.stage}:${$.name}`,$);const M=[...T.values()],X=new TextEncoder;if(M.length>256||M.reduce(($,ee)=>$+X.encode(ee.text).length,0)>32*1024*1024)throw new Error("Workspace limit exceeded. Clear the workspace before loading a new compilation.");t(M),n(!1),f(null),m($=>({...r?{}:$,...Object.fromEntries(u.slice().reverse().map(ee=>[ee.stage,ee.name]))})),u.some($=>$.stage==="clean")?p("clean"):u.some($=>$.stage==="raw")&&p("raw"),C(`Loaded ${u.length} artifact${u.length===1?"":"s"}${g?`; ${g} unsupported or empty entries skipped.`:"."}`)}catch(u){v(u instanceof Error?u.message:"Unable to read artifacts.")}finally{te.current=!1,D(!1),Q.current&&(Q.current.value="")}}}function we(a){a.dataTransfer.files.length&&(a.preventDefault(),h(0),ye(Array.from(a.dataTransfer.files)))}function ke(a,u){u.preventDefault(),u.currentTarget.setPointerCapture(u.pointerId)}function be(a,u){if(!u.currentTarget.hasPointerCapture(u.pointerId))return;const g=he.current.getBoundingClientRect(),T=a==="col"?(u.clientX-g.left)/g.width:(u.clientY-g.top)/g.height;U(M=>({...M,[a]:tt(T*100)}))}function ft(a,u){d&&i(null),a.refs.some(g=>g.stage==="raw")&&!a.refs.some(g=>g.stage==="clean")&&p("raw"),a.refs.some(g=>g.stage==="clean")&&!a.refs.some(g=>g.stage==="raw")&&p("clean"),f({refs:a.refs,finding:u,message:`${a.status==="confirmed"?"Confirmed evidence":"Potential issue"}: ${a.title}`})}function ut(){const a=b==="dark"?"light":"dark";x(a);try{localStorage.setItem("cute-explorer-theme",a)}catch{}}return c.jsxs("main",{"data-theme":b,onDrop:we,onDragEnter:a=>{a.dataTransfer.types.includes("Files")&&(a.preventDefault(),h(u=>u+1))},onDragLeave:()=>h(a=>Math.max(0,a-1)),onDragOver:a=>{a.dataTransfer.types.includes("Files")&&(a.preventDefault(),a.dataTransfer.dropEffect="copy")},children:[c.jsxs("header",{className:"topbar",children:[c.jsxs("div",{className:"brand",children:[c.jsx("span",{className:"brand-icon",children:c.jsx(_e,{name:"code"})}),c.jsxs("div",{children:[c.jsx("h1",{children:"CuTe Compiler Explorer"}),c.jsx("span",{children:"Local artifact inspector"})]})]}),c.jsxs("div",{className:"toolbar",children:[c.jsxs("button",{className:"primary",onClick:()=>{var a;return(a=Q.current)==null?void 0:a.click()},disabled:y,children:[c.jsx(_e,{name:"upload"}),y?"Reading…":"Open files"]}),c.jsx("input",{ref:Q,type:"file",multiple:!0,hidden:!0,accept:".py,.mlir,.ptx,.sass,.disasm,.s,.txt,.log,.zip",onChange:a=>void ye(Array.from(a.target.files??[]))}),c.jsx("button",{onClick:()=>me(!0),disabled:y||!Se.length,children:"Load sample"}),c.jsx("button",{onClick:()=>me(!1),disabled:y||!e.length,children:"Clear workspace"}),c.jsx("button",{"aria-expanded":l,onClick:()=>k(a=>!a),"aria-label":l?"Hide diagnostics":"Show diagnostics",children:"Diagnostics"}),c.jsx("button",{className:"icon theme",onClick:ut,title:"Toggle light/dark theme","aria-label":"Toggle theme",children:c.jsx(_e,{name:"theme"})})]})]}),c.jsxs("div",{className:"context",children:[c.jsx("span",{className:"workspace-name",children:r?"Sample kernel":e.length?"Local workspace":"Empty workspace"}),c.jsx("span",{className:"context-separator","aria-hidden":"true"}),c.jsx("span",{children:"Drop a ZIP or individual files"}),c.jsx("span",{className:"context-note",children:j||"Artifacts stay in this browser"})]}),S&&c.jsxs("div",{className:"error",role:"alert",children:[S,c.jsx("button",{onClick:()=>v(""),"aria-label":"Dismiss error",children:"Dismiss"})]}),c.jsxs("div",{className:"body-grid","data-diagnostics":l,children:[c.jsxs("div",{className:"workspace",ref:he,"data-expanded":d??"",style:{"--col":E.col/100,"--row":E.row/100},children:[ur.map(a=>{const u=a.id==="mlir"?ie:a.id;return c.jsx(dr,{pane:a,stage:u,artifact:O[u],choices:B[u]??[],pick:g=>{m(T=>({...T,[u]:g})),f(null)},expanded:d===a.id,expand:()=>i(g=>g===a.id?null:a.id),marks:K.result[u]??[],target:K.destination[u],select:g=>f({refs:[g]}),controls:a.id==="mlir"&&c.jsx("div",{className:"ir-toggle","aria-label":"MLIR version",children:["raw","clean"].map(g=>c.jsx("button",{disabled:!O[g],title:O[g]?void 0:"Artifact not supplied","aria-pressed":g===ie&&!!O[g],onClick:()=>p(g),children:g==="raw"?"Raw IR":"Clean IR"},g))})},a.id)}),["col","row"].map(a=>c.jsx("div",{className:`split ${a}`,role:"separator",tabIndex:0,"aria-label":a==="col"?"Resize columns":"Resize rows","aria-orientation":a==="col"?"vertical":"horizontal","aria-valuemin":20,"aria-valuemax":80,"aria-valuenow":Math.round(E[a]),onPointerDown:u=>ke(a,u),onPointerMove:u=>be(a,u),onKeyDown:u=>{const g=a==="col"?u.key==="ArrowRight"?3:u.key==="ArrowLeft"?-3:0:u.key==="ArrowDown"?3:u.key==="ArrowUp"?-3:0;g&&(u.preventDefault(),U(T=>({...T,[a]:tt(T[a]+g)})))}},a))]}),c.jsxs("aside",{className:"diagnostics","aria-label":"Kernel diagnostics",hidden:!l,children:[c.jsxs("header",{children:[c.jsx("h2",{children:"Kernel diagnostics"}),c.jsx("span",{className:"count",children:(L==null?void 0:L.findings.length)??"–"})]}),c.jsx("p",{className:"aside-note",children:"Rule-based observations. Confirmed means the evidence is present, not that it is a bug."}),B.report&&c.jsxs("label",{className:"report-picker",children:["Resource report",c.jsx("select",{className:"artifact-name",value:(Ue=O.report)==null?void 0:Ue.name,onChange:a=>{m(u=>({...u,report:a.target.value})),f(null)},children:B.report.map(a=>c.jsx("option",{value:a.name,children:a.name},a.name))})]}),se.length>0&&O.report&&c.jsxs("section",{ref:ve,tabIndex:-1,className:"report-evidence","aria-label":"Resource report evidence",children:[c.jsx("h3",{children:Pe(O.report.name)}),se.map(a=>{var u;return c.jsx("pre",{children:c.jsxs("mark",{children:[a.line,": ",(u=Y.get("report"))==null?void 0:u[a.line-1]]})},je(a))})]}),oe&&c.jsx("p",{className:"no-findings",children:oe}),!L&&!oe&&c.jsx("p",{className:"no-findings",children:A.length?"Analyzing artifacts…":"Load artifacts to run diagnostics."}),(L==null?void 0:L.findings.length)===0&&c.jsx("p",{className:"no-findings",children:"No findings from the current artifacts."}),c.jsx("div",{className:"finding-list",children:L==null?void 0:L.findings.map((a,u)=>{const g=`${a.kind}:${u}`,T=[...new Map(a.refs.flatMap(M=>ne.spots(M)).map(M=>[`${M.file}:${M.line}`,M])).values()];return c.jsxs("button",{className:"finding","aria-pressed":(s==null?void 0:s.finding)===g,onClick:()=>ft(a,g),children:[c.jsxs("span",{className:"finding-top",children:[c.jsx("span",{className:"finding-title",children:a.title}),c.jsx("span",{className:`badge ${a.status}`,children:a.status})]}),c.jsx("span",{className:"finding-body",children:a.explanation}),c.jsx("span",{className:"finding-impact",children:a.impact}),c.jsx("span",{className:"evidence",children:T.length?`Source: ${T.slice(0,3).map(M=>`${Pe(M.file)}:${M.line}`).join(", ")}${T.length>3?" …":""}`:"No source location in supplied evidence."}),a.refs.slice(0,4).map(M=>{var $;const X=Y.get(M.stage);return c.jsxs("span",{className:"evidence",children:[fr[M.stage],":",M.line," ",($=X==null?void 0:X[M.line-1])==null?void 0:$.trim().slice(0,160)]},je(M))})]},g)})}),L&&c.jsxs("details",{children:[c.jsx("summary",{children:"Analysis coverage"}),c.jsx("ul",{className:"coverage",children:L.coverage.map(a=>c.jsx("li",{children:a.note},a.kind))})]})]})]}),c.jsxs("footer",{children:[c.jsx("span",{className:"selection-status",role:"status",children:xe}),c.jsxs("div",{className:"legend",children:[c.jsxs("span",{children:[c.jsx("i",{className:"direct"})," Metadata"]}),c.jsxs("span",{children:[c.jsx("i",{className:"approx"})," Approximate"]})]})]}),w>0&&c.jsxs("div",{className:"drop-overlay",children:[c.jsx("span",{children:"Drop compilation artifacts"}),c.jsx("small",{children:"Python · MLIR · PTX · SASS · ZIP"})]})]})}function gr(e){return e.reduce((t,r)=>{var n;return(t[n=r.stage]??(t[n]=[])).push(r),t},{})}function tt(e){return Math.min(80,Math.max(20,e))}dt.createRoot(document.getElementById("root")).render(c.jsx(R.StrictMode,{children:c.jsx(_r,{})}));
