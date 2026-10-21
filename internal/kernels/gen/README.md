# Kernel generators

Scripts that write the longer assembly kernels of this package, run from
the package directory and pasted over the kernel they name in the
assembly file:

    python3 gen/pool.py          # poolF32 in elem_amd64.s
    python3 gen/convgather.py    # convGatherF32 in sgemm_amd64.s

Each script prints the kernel's whole TEXT block.
