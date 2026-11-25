import cutlass
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
