# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as ext
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as bw


class AsyncCompute:
    def __init__(self, mode, stages, slots):
        self.mode, self.stages, self.slots = mode, stages, slots
    @ext.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        if cutlass.const_expr(self.mode == 'tma'):
            self.tma_kernel(a, out).launch(
                grid=(cute.ceil_div(out.shape[0], 128), 1, 1),
                block=(192, 1, 1), smem=65536)
        else:
            self.kernel(a, b, out).launch(
                grid=(cute.ceil_div(out.shape[0], 128), 1, 1),
                block=(192, 1, 1), smem=147456)
    @ext.kernel
    def tma_kernel(self, a: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        s = ext.allocate(cutlass.Float16, cutlass.AddressSpace.smem,
            cute.make_layout((128, 64, self.stages), stride=(64, 1, 8192)), alignment=128)
        g = cute.local_tile(a, (128, 64), (cta, 0, None))
        pipe = ext.TMAToAsyncPipeline.create(num_stages=self.stages,
            consumer=ext.OperationTypeEnum.LD_SHARED, consumer_arv_count=128)
        if warp == 5:
            for i in cutlass.range(a.shape[2]):
                token, idx = pipe.producer_acquire_and_get_stage()
                ext.tma_load(g[None, None, i], s[None, None, idx], ext.get_mbarrier(token))
                pipe.producer_commit_and_advance()
        elif warp < 4:
            for i in cutlass.range(a.shape[2]):
                _, idx = pipe.consumer_wait_and_get_stage()
                for k in cutlass.range_constexpr(64):
                    value = cutlass.Float32(s[tid, k, idx]) * 2.0 + 1.0
                    if cta * 128 + tid < out.shape[0]:
                        out[cta * 128 + tid, k, i] = value
                pipe.consumer_release_and_advance()
    @ext.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mma = bw.make_trivial_tiled_mma(
            cutlass.Float16, cutlass.Float16, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE,
            (128, 64))
        tiler = (128, 64, 64)
        sa = ext.allocate(cutlass.Float16, cutlass.AddressSpace.smem,
                          bw.make_smem_layout_a(mma, tiler, cutlass.Float16, self.stages), alignment=1024)
        sb = ext.allocate(cutlass.Float16, cutlass.AddressSpace.smem,
                          bw.make_smem_layout_b(mma, tiler, cutlass.Float16, self.stages), alignment=1024)
        sp = ext.allocate(cutlass.Float16, cutlass.AddressSpace.smem,
                          bw.make_smem_layout_a(mma, tiler, cutlass.Float16, self.stages), alignment=1024)
        ts = ext.allocate(cutlass.Float32, cutlass.AddressSpace.tmem,
                          ext.make_tmem_layout_acc(mma, tiler, self.stages), alignment=16)
        to = ext.allocate(cutlass.Float32, cutlass.AddressSpace.tmem,
                          ext.make_tmem_layout_acc(mma, tiler, self.slots), alignment=16)
        ga = cute.local_tile(a, (128, 64), (cta, 0, None))
        gb = cute.local_tile(b, (64, 64), (0, 0, None))
        ga = mma.get_slice(0).partition_A(ga)
        gb = mma.get_slice(0).partition_B(gb)
        op = ext.OperationTypeEnum.SM100_MMA_1SM_SS
        ab = ext.TMAToUMMAPipeline.create(num_stages=self.stages, mma_operation_type=op)
        s_pipe = ext.UMMAtoAsyncPipeline.create(num_stages=self.stages,
            mma_operation_type=op, consumer=ext.OperationTypeEnum.SM100_COPY_T2R, consumer_arv_count=128)
        p_pipe = ext.AsyncToUMMAPipeline.create(num_stages=self.stages,
            producer=ext.OperationTypeEnum.SM100_COPY_R2T, producer_arv_count=128, mma_operation_type=op)
        o_pipe = ext.UMMAtoAsyncPipeline.create(num_stages=self.slots,
            mma_operation_type=op, consumer=ext.OperationTypeEnum.SM100_COPY_T2R, consumer_arv_count=128)
        ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), cutlass.Float32)
        st = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32, tcgen05.Unpack.NONE), cutlass.Float32)
        plane = ts[(None, None), 0, 0, 0]
        t2r = tcgen05.make_tmem_copy(ld, plane).get_slice(tid % 128)
        r2t = tcgen05.make_tmem_copy(st, plane).get_slice(tid % 128)
        r = ext.allocate(cutlass.Float32, cutlass.AddressSpace.rmem, t2r.partition_D(plane).layout, alignment=32)
        coords = t2r.partition_D(cute.make_identity_tensor((128, 64)))
        atom = cute.make_mma_atom(mma.op)
        if warp == 5:
            for i in cutlass.range(a.shape[2]):
                token, idx = ab.producer_acquire_and_get_stage()
                bar = ext.get_mbarrier(token)

                ext.tma_load(ga[None, None, None, i], sa[None, None, None, idx], bar)
                ext.tma_load(gb[None, None, None, i], sb[None, None, None, idx], bar)
                ab.producer_commit_and_advance()
        elif warp == 4:
            for i in cutlass.range(a.shape[2]):
                _, ai = ab.consumer_wait_and_get_stage()
                _, si = s_pipe.producer_acquire_and_get_stage()
                atom.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(4):
                    ext.dot(atom, sa[None, None, k, ai], sb[None, None, k, ai], ts[None, None, None, si])
                    atom.set(tcgen05.Field.ACCUMULATE, True)
                s_pipe.producer_commit_and_advance()
                if cutlass.const_expr(self.mode != 'gemm'):
                    _, pi = p_pipe.consumer_wait_and_get_stage()
                    _, oi = o_pipe.producer_acquire_and_get_stage()
                    atom.set(tcgen05.Field.ACCUMULATE, i >= self.slots if self.mode == 'feedback' else False)
                    for k in cutlass.range_constexpr(4):
                        ext.dot(atom, sp[None, None, k, pi], sb[None, None, k, ai], to[None, None, None, oi])
                        atom.set(tcgen05.Field.ACCUMULATE, True)
                    p_pipe.consumer_release_and_advance()
                    o_pipe.producer_commit_and_advance()
                ab.consumer_release_and_advance()
        elif warp < 4:
            for i in cutlass.range(a.shape[2]):
                _, si = s_pipe.consumer_wait_and_get_stage()
                ext.partition_and_copy(t2r, ts[(None, None), 0, 0, si], r)
                cute.arch.fence_view_async_tmem_load()
                s_pipe.consumer_release_and_advance()
                if cutlass.const_expr(self.mode != 'gemm'):
                    _, pi = p_pipe.producer_acquire_and_get_stage()

                    for j in cutlass.range_constexpr(cute.size(r)):
                        m, k = coords[j]
                        sp[(m, k % 16), 0, k // 16, pi] = cutlass.Float16(r[j] * 0.125 + 0.03125)
                    cute.arch.fence_view_async_shared()
                    p_pipe.producer_commit_and_advance()
                    _, oi = o_pipe.consumer_wait_and_get_stage()
                    ext.partition_and_copy(t2r, to[(None, None), 0, 0, oi], r)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(self.mode == 'feedback'):
                        r.store(r.load() * (0.5 + 0.125 * (i % 3)))

                        ext.partition_and_copy(r2t, r, to[(None, None), 0, 0, oi])
                        cute.arch.fence_view_async_tmem_store()
                    o_pipe.consumer_release_and_advance()
                for j in cutlass.range_constexpr(cute.size(r)):
                    m, n = coords[j]
                    row = cta * 128 + m
                    if row < out.shape[0]:
                        out[row, n, i] = r[j]
