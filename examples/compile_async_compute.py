import importlib.util
import os
import sys
from pathlib import Path


def compile_for_explorer():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor

    default = Path(__file__).with_name('02_async_compute.py')
    source = Path(os.environ.get('ASYNC_COMPUTE_SOURCE', default)).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('async_compute_workset_solution', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows, steps = 129, 9

    def tensor(dtype, m):
        return make_fake_tensor(dtype, (m, 64, steps), stride=(64, 1, m * 64), assumed_align=16)

    return cute.compile(
        module.AsyncCompute('feedback', 2, 2),
        tensor(cutlass.Float16, rows), tensor(cutlass.Float16, 64), tensor(cutlass.Float32, rows),
        no_jit_engine=True,
    )
