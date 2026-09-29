"""In-place multi-lane progress display for long frasci solves.

Hand-rolled rather than tqdm-based, for three reasons that all held when this
was written (2026-08-29): tqdm is not installed in FRASCIenv and this needs no
new dependency to do what a ``\\r`` write does; tqdm's multi-bar ``position=``
is the fragile part of tqdm and the concurrent-fragment case is exactly what we
need most; and no bar library can animate on a stream that is not a TTY, which
is how these runs are usually launched (nohup, redirect to a log). See
docs/prompts/frasci_library_PROMPT.md sec 5.5.

Four states behind one interface, chosen by ``enabled``:

* ``'auto'`` + TTY  -> ANSI in-place: the headline's bar line plus one chip line
  for everything else, rewritten on every repaint, throttled to ~10 Hz. This is
  the "one smooth bar" case.
* ``'auto'`` + Jupyter kernel -> the same frame published once with an IPython
  display ID, then replaced in place with ``update_display``. No widgets needed.
* ``'auto'`` + no TTY -> a periodic ``\\n``-terminated one-line digest every
  ``interval`` seconds (30 s by default). A log file gets a readable timeline,
  not thousands of near-duplicate bar snapshots.
* ``False`` -> a no-op object. Every call site can stay unconditional.

One bar, not one per lane
-------------------------
A caller may nominate one lane as the ``headline=``. That lane gets the bar and
the full line; every other lane collapses to a bar-free "chip" -- key, phase,
seconds in the current step, and whatever fields the call site set (energy).
This exists because per-fragment ``done/total`` bars were actively misleading:
TrimCI's growth loop overshoots its target, so a fragment's bar pins at 100%
("104/100") early in a macro cycle and then sits there for the remaining 40-70 s
of real compute, reading as "nothing is happening". A chip that shows a phase, a
live second count and a moving energy carries the same information without
pretending to measure completion. With no ``headline=`` every lane renders in
full, which is what a two-or-three-step pipeline wants.

Honesty rule for the bar fill
-----------------------------
A lane's fill fraction is only ever ``done/total`` for a total that is genuinely
known ahead of time -- determinants against a configured cap, macro cycle i of
``max_cycle_macro``, row i of N in a fixed-length loop. A soft target the loop
is allowed to overshoot is not a denominator; show the raw count as a field
instead. ``total <= 0`` renders as blank space, never as a fabricated fill.
"""
from __future__ import annotations

import contextlib
import re
import sys
import threading
import time

_BAR_W = 22
_MIN_REPAINT_S = 0.1

# mrh's LASSCF macro-cycle line, emitted by pyscf's logger at las.verbose >= 4:
#   LASSCF macro 3 : E = -549.6738 ; |g_int| = 1.2e-04 ; |g_ci| = ... ; |g_x| = ...
_MACRO_RE = re.compile(
    r"LASSCF macro (\d+) : E = (\S+) ; \|g_int\| = (\S+)")


def _in_jupyter_kernel():
    """True only in a Jupyter kernel, not plain IPython or a normal script."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return (shell is not None
            and shell.__class__.__name__ == "ZMQInteractiveShell")


def _notebook_display_functions():
    # Imported only after _in_jupyter_kernel() succeeds: IPython remains an
    # optional dependency for every terminal/script caller.
    from IPython.display import display, update_display
    return display, update_display


def _fmt_bar(done: int, total: int) -> str:
    frac = 0.0 if total <= 0 else min(1.0, done / total)
    filled = int(round(_BAR_W * frac))
    return "#" * filled + "." * (_BAR_W - filled)


class _Lane:
    __slots__ = ("key", "phase", "done", "total", "fields", "t0", "t_end")

    def __init__(self, key):
        self.key = key
        self.phase = "initial CI"
        self.done = 0
        self.total = 0
        self.fields = {}
        self.t0 = None
        self.t_end = None

    def _elapsed(self, now):
        return 0.0 if self.t0 is None else (self.t_end or now) - self.t0

    def render(self, now):
        if self.total > 0:
            bar = _fmt_bar(self.done, self.total)
            amount = f" {self.done:>7}/{self.total:<7}"
        else:
            bar, amount = " " * _BAR_W, ""
        extra = "  ".join(f"{k}={v}" for k, v in self.fields.items())
        # Truncate, never overflow: one long phase label pushing its line wider
        # than the others is what makes a multi-lane block look ragged.
        return (f"{self.key[:10]:<10} {self.phase[:13]:<13} [{bar}]{amount} "
                f"{self._elapsed(now):6.1f}s  {extra}")

    def chip(self, now):
        """Bar-free one-liner: what this lane is doing and how long it has been
        doing it. Fields come from the call site verbatim."""
        parts = [self.key, self.phase[:10], f"{self._elapsed(now):.0f}s"]
        parts += [f"{k}={v}" for k, v in self.fields.items()]
        return " ".join(parts)


class Progress:
    """A set of named lanes rendered together. Thread-safe: the parallel
    fragment dispatch updates lanes from several dispatcher threads at once.
    """

    def __init__(self, keys, *, enabled="auto", interval=30.0, stream=None,
                 max_lanes=8, title=None, headline=None):
        self.stream = stream if stream is not None else sys.stdout
        in_notebook = bool(enabled) and _in_jupyter_kernel()
        if enabled == "auto":
            if in_notebook:
                self.tty = False
                self.mode = "notebook"
            else:
                enabled = bool(getattr(self.stream, "isatty", lambda: False)())
                self.tty = enabled
                # Not a TTY is not "off": fall back to the periodic digest,
                # which is what a nohup'd run or a redirected log wants.
                self.mode = "tty" if enabled else "periodic"
        elif enabled:
            self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
            self.mode = ("notebook" if in_notebook else
                         ("tty" if self.tty else "periodic"))
        else:
            self.tty = False
            self.mode = "off"
        self._display = self._update_display = self._display_id = None
        if self.mode == "notebook":
            self._display, self._update_display = _notebook_display_functions()
        keys = list(keys)
        self._headline = headline if headline in keys else None
        rest = [k for k in keys if k != self._headline]
        # ponytail: collapse to a single aggregate lane past max_lanes rather
        # than scrolling a 64-fragment (H2)n partition off the screen. The
        # headline is exempt -- it is the one signal that must survive. Upgrade
        # path if per-fragment detail is wanted at that size: a scrolling window
        # of the N slowest lanes.
        self.collapsed = len(rest) > max_lanes
        self._keys = ([self._headline] if self._headline else []) + (
            ["all"] if self.collapsed else rest)
        self._lanes = {k: _Lane(k) for k in self._keys}
        self._lock = threading.Lock()
        self._interval = float(interval)
        self._last_paint = 0.0
        self._painted = 0
        self._title = title

    def _map(self, key):
        if key == self._headline or not self.collapsed:
            return key
        return "all"

    def lane(self, key):
        return self._lanes[self._map(key)]

    def update(self, key, *, phase=None, done=None, total=None,
               reset_clock=False, finished=False, **fields):
        if self.mode == "off":
            return
        with self._lock:
            lane = self._lanes[self._map(key)]
            now = time.monotonic()
            if lane.t0 is None or reset_clock:
                lane.t0 = now
                lane.t_end = None
            if phase is not None:
                lane.phase = phase
            if done is not None:
                lane.done = int(done)
            if total is not None:
                lane.total = int(total)
            lane.t_end = now if finished else None
            lane.fields.update(fields)
        self._paint()

    def write(self, text):
        """Emit a line above the bars without corrupting them."""
        text = text.rstrip("\n")
        if not text:
            return
        if self.mode == "off":
            return
        with self._lock:
            if self.mode == "tty":
                self._clear()
            self.stream.write(text + "\n")
            self.stream.flush()
        if self.mode == "tty":
            self._paint(force=True)

    def _clear(self):
        if self._painted:
            self.stream.write(f"\x1b[{self._painted}A\x1b[J")
            self._painted = 0

    def _lines(self, now):
        """The body of a frame: one bar line for the headline (or for every
        lane when there is no headline), plus one chip line for the rest."""
        if self._headline is None:
            return [lane.render(now) for lane in self._lanes.values()]
        out = [self._lanes[self._headline].render(now)]
        chips = "   ".join(lane.chip(now) for key, lane in self._lanes.items()
                           if key != self._headline)
        if chips:
            out.append(chips)
        return out

    def _paint(self, force=False):
        now = time.monotonic()
        if self.mode in ("tty", "notebook"):
            if not force and now - self._last_paint < _MIN_REPAINT_S:
                return
            with self._lock:
                self._last_paint = now
                lines = self._lines(now)
                if self._title:
                    lines.insert(0, self._title)
                if self.mode == "tty":
                    self._clear()
                    self.stream.write("\n".join(lines) + "\n")
                    self.stream.flush()
                    self._painted = len(lines)
                else:
                    frame = {"text/plain": "\n".join(lines)}
                    if self._display_id is None:
                        handle = self._display(
                            frame, raw=True, display_id=True)
                        self._display_id = handle.display_id
                    else:
                        self._update_display(
                            frame, raw=True, display_id=self._display_id)
        elif self.mode == "periodic":
            if not force and now - self._last_paint < self._interval:
                return
            with self._lock:
                self._last_paint = now
                digest = " | ".join(x.strip() for x in self._lines(now))
                self.stream.write(digest + "\n")
                self.stream.flush()

    @contextlib.contextmanager
    def capture(self):
        """Route ``sys.stdout``/``sys.stderr`` through this display for the
        duration (TTY mode only).

        Not optional polish: a repainting bar owns a fixed block of terminal
        lines, so ANY other writer to that terminal desynchronises the
        cursor-up count and the display starts duplicating itself -- which is
        the "new bar every time" symptom. pyscf/mrh write warnings to stderr,
        a stream the bar does not own, so without this they corrupt it (50 of
        148 frames in the first alfe2 measurement). One screen needs one writer.

        Left alone in periodic mode: interleaved warnings in a log file are
        useful and cannot corrupt anything.
        """
        if self.mode != "tty":
            yield
            return
        saved = sys.stdout, sys.stderr
        sys.stdout = MacroTap(self, None, None, echo=True)
        sys.stderr = MacroTap(self, None, None, echo=True)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = saved

    def close(self):
        if self.mode == "off":
            return
        self._paint(force=True)
        if self.mode == "tty":
            self._painted = 0  # leave the final frame on screen

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class NullProgress:
    """No-op stand-in so call sites never branch on progress being enabled."""
    mode = "off"
    collapsed = False

    def update(self, *a, **k):
        pass

    def write(self, text):
        pass

    def close(self):
        pass

    @contextlib.contextmanager
    def capture(self):
        yield

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make(keys, progress, **kwargs):
    """``progress`` is False/None -> NullProgress, else a Progress."""
    if not progress:
        return NullProgress()
    return Progress(keys, enabled=("auto" if progress is True else progress),
                    **kwargs)


class MacroTap:
    """A file-like stand-in for ``las.stdout``.

    Two jobs, and the second is why this exists at all: a ``\\r``-animated bar
    cannot share a stream with something else that prints, so once a bar owns
    stdout mrh's logger output HAS to be routed through the bar's own
    ``write()``. Having intercepted it anyway, parsing ``|g_int|`` out of the
    macro-cycle line is a one-line regex -- and that line is the only place
    LASSCF's convergence signal is exposed (mrh has no user callback hook).
    """

    def __init__(self, progress, key, max_cycle, *, echo=False):
        self.progress = progress
        # key=None makes this a plain line-buffered forwarder into the display
        # (what Progress.capture() installs over sys.stdout/sys.stderr).
        self.key = key
        # las.max_cycle_macro is a hard stop, so macro/max_cycle is a real
        # denominator -- an upper bound on the work, since convergence usually
        # trips first. That overestimate is the price of a bar that always
        # moves; |g_int| beside it is the actual convergence signal.
        self.max_cycle = None if max_cycle is None else int(max_cycle)
        self.echo = echo
        self._buf = ""
        self.macro = 0
        self.g_int = None
        self.e_tot = None

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._line(line)
        return len(text)

    def _line(self, line):
        m = _MACRO_RE.search(line) if self.key is not None else None
        if m:
            self.macro = int(m.group(1))
            self.e_tot = float(m.group(2))
            self.g_int = float(m.group(3))
            self.progress.update(
                self.key, phase="macro cycle",
                done=self.macro + 1, total=self.max_cycle or 0,
                E=f"{self.e_tot:.6f}", g_int=f"{self.g_int:.2e}")
        elif self.echo and line.strip():
            self.progress.write(line)

    def flush(self):
        # suppress_fd() flushes sys.stdout before dup2'ing fd 1 to /dev/null,
        # to keep pending bytes out of the void; forward so that still holds
        # while this stands in for the real stream.
        getattr(self.progress, "stream", None) and self.progress.stream.flush()

    def isatty(self):
        return False

    def writable(self):
        return True


def demo():
    """Self-check: the honesty rules and both renderers, no solve required."""
    import io

    assert _fmt_bar(0, 100).count("#") == 0
    assert _fmt_bar(100, 100).count("#") == _BAR_W
    assert _fmt_bar(200, 100).count("#") == _BAR_W, "fill must clamp at 100%"
    assert _fmt_bar(50, 0).count("#") == 0, "unknown total must not fabricate fill"

    # Non-TTY must never emit a carriage return, and must be quiet between ticks.
    buf = io.StringIO()
    p = Progress(["F0", "F1"], enabled=True, stream=buf, interval=1e6)
    p.update("F0", phase="phase1", done=5, total=10)
    p.update("F1", phase="phase1", done=1, total=10)
    p.close()
    out = buf.getvalue()
    assert p.mode == "periodic", p.mode
    assert "\r" not in out, "non-TTY output must be newline-terminated only"
    assert out.count("\n") == 1, f"expected one digest line, got {out!r}"
    assert "F0" in out and "F1" in out

    # Many lanes collapse instead of scrolling the terminal away.
    p2 = Progress([f"F{i}" for i in range(64)], enabled=True, stream=io.StringIO(),
                  max_lanes=8)
    assert p2.collapsed and list(p2._lanes) == ["all"]
    p2.update("F17", phase="phase1", done=3, total=9)  # must not KeyError

    # ...but a headline lane survives the collapse, since it is the one signal
    # the whole display exists to show.
    p2h = Progress(["las"] + [f"F{i}" for i in range(64)], enabled=True,
                   stream=io.StringIO(), max_lanes=8, headline="las")
    assert p2h.collapsed and list(p2h._lanes) == ["las", "all"]

    # With a headline, exactly ONE lane draws a bar; the rest are chips. This is
    # the whole point of the redesign -- four concatenated bars read as noise.
    buf2 = io.StringIO()
    p5 = Progress(["las", "F0", "F1", "F2"], enabled=True, stream=buf2,
                  interval=1e6, headline="las")
    p5.update("las", phase="macro cycle", done=2, total=50,
              E="-326.975584", g_int="2.21e-02")
    for i, e in enumerate((-23.855421, -42.517852, -56.510225)):
        p5.update(f"F{i}", phase="refine", E=f"{e:.6f}", n=104)
    p5.close()
    line = buf2.getvalue()
    bars_drawn = re.compile(r"\[[#. ]{%d}\]" % _BAR_W).findall
    assert line.count("\n") == 1, "periodic mode stays one line per tick"
    assert len(bars_drawn(line)) == 1, f"expected a single bar, got: {line}"
    assert "2/50" in line, "headline must show macro cycle against its cap"
    for frag in ("F0", "F1", "F2"):
        assert frag in line, f"{frag} missing from chip strip"
    # Energies are the part the display is actually read for; keep them all.
    for e in ("-326.975584", "-23.855421", "-42.517852", "-56.510225"):
        assert e in line, f"{e} missing from digest"

    # No headline -> every lane renders in full, i.e. frasci.recovery unchanged.
    buf2b = io.StringIO()
    p6 = Progress(["phase01", "phase2"], enabled=True, stream=buf2b, interval=1e6)
    p6.update("phase01", phase="discovery+COO", done=5, total=10)
    p6.update("phase2", phase="frozen growth", done=1, total=10)
    p6.close()
    assert len(bars_drawn(buf2b.getvalue())) == 2

    # MacroTap turns mrh's log line into a lane update and swallows the rest.
    class _Rec:
        mode = "periodic"

        def __init__(self):
            self.calls = []
            self.lines = []

        def update(self, key, **kw):
            self.calls.append((key, kw))

        def write(self, text):
            self.lines.append(text)

    rec = _Rec()
    tap = MacroTap(rec, "las", 50)
    tap.write("LASSCF macro 3 : E = -549.673872 ; |g_int| = 1.5e-02 ;"
              " |g_ci| = 1e-9 ; |g_x| = 1e-9\n")
    tap.write("some other mrh chatter\n")
    assert tap.macro == 3 and tap.g_int == 1.5e-2
    assert len(rec.calls) == 1 and rec.calls[0][0] == "las"
    assert rec.calls[0][1]["done"] == 4 and rec.calls[0][1]["total"] == 50
    assert rec.calls[0][1]["g_int"] == "1.50e-02", "g_int stays visible as text"
    assert rec.lines == [], "echo=False must not forward unmatched lines"

    # Unknown cap must leave the bar blank rather than invent a denominator.
    rec2 = _Rec()
    MacroTap(rec2, "las", None).write(
        "LASSCF macro 1 : E = -1.0 ; |g_int| = 1e-2 ; |g_ci| = 0 ; |g_x| = 0\n")
    assert rec2.calls[0][1]["total"] == 0

    # capture() must send BOTH stdout and stderr through the display in TTY
    # mode -- an uncaptured stderr writer is what desynchronises the repaint.
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    tty = _Tty()
    p3 = Progress(["F0"], enabled="auto", stream=tty)
    assert p3.mode == "tty"
    saved = sys.stdout, sys.stderr
    with p3.capture():
        assert sys.stdout is not saved[0] and sys.stderr is not saved[1]
        print("via stdout")
        print("WARN: via stderr", file=sys.stderr)
    assert (sys.stdout, sys.stderr) == saved, "capture must restore the streams"
    p3.close()
    got = tty.getvalue()
    assert "via stdout" in got and "WARN: via stderr" in got
    assert "\x1b[" in got, "TTY mode must emit in-place cursor moves"

    # ...and must leave the streams alone when there is no bar to corrupt.
    p4 = Progress(["F0"], enabled=True, stream=io.StringIO())
    with p4.capture():
        assert (sys.stdout, sys.stderr) == saved

    # Disabled progress writes nothing at all.
    buf3 = io.StringIO()
    n = make(["F0"], False)
    n.update("F0", phase="x", done=1, total=2)
    with n.capture():
        pass
    n.close()
    assert isinstance(n, NullProgress) and buf3.getvalue() == ""

    print("frasci._progress demo: OK")


if __name__ == "__main__":
    demo()
