from __future__ import annotations

import os
import sys

from . import adamixture as _impl


def main() -> int | None:
    # `jx` removes the top-level module name before invoking this function, so
    # the first remaining token is the FastPop subcommand.  Keep the legacy
    # `jx fastpop -bfile ...` form as an implicit structure invocation.
    if len(sys.argv) > 1:
        subcommand = str(sys.argv[1]).strip().lower()
        if subcommand in {"fst", "xpclr"}:
            del sys.argv[1]
            if subcommand == "fst":
                from . import fastpop_fst

                return fastpop_fst.main()
            from . import fastpop_xpclr

            return fastpop_xpclr.main()
        if subcommand == "structure":
            del sys.argv[1]
    os.environ["JANUSX_POPSTRUCT_BRAND"] = "fastpop"
    result = _impl.main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers

    install_interrupt_handlers()
    main()
