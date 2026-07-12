# AGENTS.md

Instructions for AI agents writing code that uses pypelite.

## Canonical Style

- Always call decorators: use `@pypelite.checkpoint()` and
  `@pypelite.stage()`. Never use a bare decorator.
- Omit the decorator name unless the archive name must differ from the Python
  function name.
- When a custom name is required, pass it as `name=...`; positional decorator
  names are not supported.
- Use `checkpoint()` for one artifact per pipeline run step.
- Use `stage()` for one artifact per call identity. All arguments form the key
  by default.
- Prefer ordinary Python control flow inside one `pypelite.pipeline(...)`
  context. Do not construct a parallel DAG abstraction.
- Keep archive and control names short and unqualified. Pypelite handles
  process-global registry tie-breaking internally.
- Pass an ordinary default archive path positionally to `pipeline(...)`. Use
  `archive=` only for a configured `pypelite.Archive` and `archives=` only for
  additional named archives.
- Put shared formatters on the default archive and overrides on a named
  archive. Resolution checks the named archive, the default archive, then
  pickle.
- Use `pypelite.argument_parser()` for command-line run controls instead of
  defining parallel refresh, clean, skip, until, or read-only flags.

## Identity and Parallelism

- Accept the default stage key unless a smaller stable identity is part of the
  data contract.
- Dictionary keys and set members are order-independent; equal dictionaries
  or sets share a cache identity.
- When needed, select a key with a parameter name, tuple of parameter names,
  or callable, in that order of preference.
- Use `vectorize` when the decorated function accepts one item and callers pass
  a collection; pypelite maps the function over missing items.
- Use `batch` when the decorated function itself accepts a collection;
  pypelite passes each worker a collection of missing items.
- Batched stages must return one output per input item. Do not filter, expand,
  or aggregate their result list.
- Both stage forms cache one artifact per item. A batched checkpoint instead
  concatenates worker results into its one artifact.
- Add `workers` only after choosing `vectorize` or `batch`.
- Omit `batch_size` for automatic sizing and `workers` for one worker.
  Explicit values must be positive integers; do not pass booleans, zero,
  negative values, or numeric strings.
- Name the `vectorize` or `batch` parameter explicitly; boolean shorthand is
  not supported.
- Use `source=True` only when source changes should intentionally create new
  artifacts.
- Cache hits reject changed non-key arguments by default. Key fields and the
  item consumed by a vectorized or batched stage define cache identity and
  are not compared again. Checkpoints compare every argument because they
  have no key.
- Use `reject_changed=False` only when changed non-key arguments should also
  reuse the cached result.
- With `reject_changed=False`, non-key arguments are stored only as readable
  audit strings capped at 1000 characters. Otherwise they are hashed for
  comparison. Hashed values and formatter names are deduplicated in the
  metadata value table.
- Treat `StageMetadata` as the exact archive metadata contract. Do not add
  fields, optional representations, fallbacks, or migrations without updating
  its validator and the library version.
- Catch `pypelite.PypeliteError` only when the caller can handle metadata or
  cache contract failures; otherwise let it propagate.

## Review

- Generated pipeline code should remain short enough to audit directly.
- Prefer library defaults over spelling out default options.
- Do not add compatibility branches, cache migrations, or defensive fallbacks
  unless the user explicitly requests them.
- Document the visible archive layout and any non-default cache identity.
