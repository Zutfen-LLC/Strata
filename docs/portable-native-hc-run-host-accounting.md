# Composed HC run host payload

The standalone HC composition admission conservatively sums the following source-controlled vector payloads, even though the component preflights run serially. Rows are restricted to one or two.

* Pending: one-row residual, block and injection downloads, plus the one-row control residual, gate and exponential outputs: `(10240+2560+4+10240+4+4)*4` bytes. The loop reuses inputs across rows.
* Norm/injection: residual download and oracle normalized output, each `rows*10240` floats; norm weights `10240` floats, injection weights `40960` BF16 values, and oracle rs and injection outputs each `rows*4` floats.
* Mix: one temporary scan vector at a time; either packed weight is 6,553,600 bytes, larger than the one/two-row input scan.

PLE token and residual copies may remain live during these dispatches, so the composition adds rather than substitutes these bounds. The existing publication/read and metadata terms remain charged. The vector sizes assume the source-requested payload capacities; allocator metadata and slack, stack storage, external backend allocations, caller-owned copies and retained reads are excluded. Injected dispatch and consumer callbacks are trusted and are not sandboxed.

This is not complete combined production accounting: pre-gate and other component work, including separate factory/rewrite accounting where applicable, still require their own admission. The standalone composition is not wired into model admission.
