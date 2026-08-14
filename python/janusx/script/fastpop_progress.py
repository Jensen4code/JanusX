"""Low-overhead terminal progress for the FastPop Rust scans."""

from __future__ import annotations

from collections.abc import Sequence

from ._common.progress import ProgressAdapter, stdout_is_tty


class FastPopProgress:
    """Render coarse Rust scan progress without entering the SNP hot loop.

    The Rust backend reports ``(stage, done, total)`` only at BED block
    boundaries.  Progress is disabled entirely for non-interactive output so
    batch jobs and redirected logs do not pay callback or terminal costs.
    """

    def __init__(
        self,
        *,
        description: str,
        stages: int = 1,
        stage_labels: Sequence[str] | None = None,
        log_unit: str = "site",
    ) -> None:
        self.enabled = bool(stdout_is_tty())
        self._stages = max(1, int(stages))
        labels = tuple(str(label) for label in (stage_labels or (description,)))
        self._labels = labels if labels else (str(description),)
        self._bar = None
        if self.enabled:
            try:
                self._bar = ProgressAdapter(
                    total=1,
                    desc=str(description),
                    show_spinner=True,
                    show_postfix=False,
                    keep_display=False,
                    show_remaining=True,
                    emit_done=False,
                    force_animate=True,
                    log_unit=str(log_unit),
                )
            except Exception:
                self.enabled = False
        self._last_stage = -1
        self._last_done = 0

    def _label(self, stage: int) -> str:
        index = min(max(0, int(stage)), len(self._labels) - 1)
        return self._labels[index]

    def callback(self, stage: int, done: int, total: int) -> None:
        """Consume a Rust callback; terminal failures never abort analysis."""

        if not self.enabled or self._bar is None:
            return
        try:
            stage_i = min(max(0, int(stage)), self._stages - 1)
            total_i = max(1, int(total))
            global_total = total_i * self._stages
            if self._last_stage != stage_i:
                self._bar.set_desc(self._label(stage_i))
                self._bar.set_total(global_total)
                self._last_stage = stage_i
            elif self._bar.total != global_total:
                self._bar.set_total(global_total)

            global_done = min(
                global_total,
                stage_i * total_i + max(0, int(done)),
            )
            delta = global_done - self._last_done
            if delta > 0:
                self._bar.update(delta)
                self._last_done = global_done
        except Exception:
            # A broken/closed terminal must not change the analysis result.
            self.enabled = False

    def finish(self) -> None:
        if self._bar is None:
            return
        try:
            self._bar.finish()
            self._bar.close(show_done=False)
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if self._bar is None:
            return
        try:
            self._bar.close(show_done=False)
        except Exception:
            self.enabled = False
