"""Opinionated pipeline run archives for Python scripts."""

__version__ = "0.1.12"

import argparse
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import fcntl
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import pathlib
import pickle
import re
import shutil
import tempfile
import types
import typing

_ACTIVE_PIPELINE = contextvars.ContextVar("pypelite_active_pipeline")
_AUDIT_LIMIT = 1000


def _digest(value):
    value = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(value).hexdigest()


@contextlib.contextmanager
def _transactional_path(path):
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=path.suffix,
    ) as f:
        temp_path = pathlib.Path(f.name)
    try:
        yield temp_path
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


class ArchiveFormat:
    """Base class for custom artifact formats.

    Subclass this when a value should not be pickled. Set ``value_type`` to the
    type the formatter handles, choose a short ``name``, and keep that name
    stable after writing artifacts.
    """

    name = "pickle"
    value_type = object
    suffix = ".pkl"

    def load(self, path):
        """Read an artifact.

        Parameters
        ----------
        path
            Artifact path using this formatter's ``suffix``.
        """

        return pickle.loads(path.read_bytes())

    def dump(self, path, value):
        """Write an artifact.

        Parameters
        ----------
        path
            Destination path. Pypelite supplies a temporary path and commits
            it transactionally after this method returns.
        value
            Python value handled by this formatter's ``value_type``.
        """

        with pathlib.Path(path).open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

    def hash(self, value):
        """Hash a value using this format.

        Parameters
        ----------
        value
            Value whose serialized content selects cache identity.

        Returns
        -------
        str
            Lowercase SHA-256 digest. Dictionaries and sets are
            order-independent.
        """

        if isinstance(value, dict):
            items = sorted(
                (self.hash(key), self.hash(item)) for key, item in value.items()
            )
            return _digest("dict:" + json.dumps(items))
        if isinstance(value, (set, frozenset)):
            hashes = sorted(self.hash(item) for item in value)
            return _digest("set:" + json.dumps(hashes))
        if isinstance(value, (list, tuple)):
            prefix = "list:" if isinstance(value, list) else "tuple:"
            return _digest(prefix + json.dumps([self.hash(x) for x in value]))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / f"value{self.suffix}"
            self.dump(path, value)
            return _digest(path.read_bytes())

    def concat(self, values):
        """Join outputs from ``batch`` workers.

        Parameters
        ----------
        values
            Worker results in submission order.

        Returns
        -------
        object
            Combined checkpoint result. The pickle implementation flattens
            one collection level.
        """

        return list(itertools.chain.from_iterable(values))


class Archive:
    """Archive root plus optional formatters.

    Formatters are tried in order. During a pipeline, additional archives
    inherit the default archive's formatters and every archive falls back to
    pickle.

    Parameters
    ----------
    path
        Directory containing cached-function subdirectories. It is created
        immediately.
    formatters
        Ordered iterable of :class:`ArchiveFormat` instances. The first
        formatter whose ``value_type`` matches a value is used.
    """

    def __init__(self, path, formatters=None):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.formatters = list(formatters or [])

    def lock(self, read_only):
        lock_file = (self.path / ".pypelite.lock").open("a+b")
        fcntl.flock(lock_file, fcntl.LOCK_SH if read_only else fcntl.LOCK_EX)
        return lock_file

    def formatter_for(self, obj):
        return next(f for f in self.formatters if isinstance(obj, f.value_type))

    def formatter_named(self, name):
        return next(f for f in self.formatters if f.name == name)


class PypeliteError(ValueError):
    """Base class for pypelite contract errors."""


def _string_map(value):
    return isinstance(value, dict) and all(
        isinstance(name, str) and isinstance(item, str)
        for name, item in value.items()
    )


def _hash_string(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _namespace(data, **field_types):
    if (
        not isinstance(data, dict)
        or set(data) != set(field_types)
        or any(
            not isinstance(data[name], value_type)
            for name, value_type in field_types.items()
        )
    ):
        raise PypeliteError("invalid metadata format")
    return types.SimpleNamespace(**data)


def _validate_metadata(data):
    metadata = _namespace(
        data,
        version=str,
        qualname=str,
        source=(str, type(None)),
        artifacts=dict,
        values=dict,
    )
    if metadata.version != __version__ or not metadata.qualname:
        raise PypeliteError("invalid metadata format")
    del metadata.version
    for value_hash, value in metadata.values.items():
        value = _namespace(value, display=str, format=str)
        if not _hash_string(value_hash) or len(value.display) > _AUDIT_LIMIT:
            raise PypeliteError("invalid metadata format")
        metadata.values[value_hash] = value
    for identity, artifact in metadata.artifacts.items():
        artifact = _namespace(
            artifact,
            arguments=dict,
            argument_hashes=dict,
            arguments_hash=(str, type(None)),
            key=dict,
            key_hash=(str, type(None)),
            file=str,
            format=str,
        )
        if (
            not _hash_string(identity)
            or not _string_map(artifact.arguments)
            or not _string_map(artifact.argument_hashes)
            or not _string_map(artifact.key)
            or any(
                len(value) > _AUDIT_LIMIT
                for value in artifact.arguments.values()
            )
        ):
            raise PypeliteError("invalid metadata format")
        references = {
            **artifact.argument_hashes,
            **artifact.key,
        }
        if any(
            not _hash_string(value_hash) or value_hash not in metadata.values
            for value_hash in references.values()
        ):
            raise PypeliteError("invalid metadata format")
        if artifact.arguments_hash is not None and artifact.arguments_hash != (
            _digest(json.dumps(artifact.argument_hashes, sort_keys=True))
        ):
            raise PypeliteError("invalid metadata format")
        if artifact.key_hash is not None and artifact.key_hash != _digest(
            json.dumps(artifact.key, sort_keys=True)
        ):
            raise PypeliteError("invalid metadata format")
        metadata.artifacts[identity] = artifact
    return metadata


@dataclasses.dataclass
class StageMetadata:
    """Metadata contract for one cached function."""

    root: pathlib.Path
    qualname: str
    source: str | None = None
    artifacts: dict = dataclasses.field(default_factory=dict)
    values: dict = dataclasses.field(default_factory=dict)
    touched: set = dataclasses.field(default_factory=set)

    @classmethod
    def load(cls, root, function):
        path = root / "meta.json"
        if not path.exists():
            return cls(root, function.qualname)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise PypeliteError("invalid metadata format") from error
        metadata = cls(root=root, **vars(_validate_metadata(data)))
        metadata.check(function)
        return metadata

    def check(self, function):
        if self.qualname != function.qualname:
            raise PypeliteError(
                f"stage name {function.name!r} belongs to "
                f"{self.qualname!r}, not {function.qualname!r}"
            )

    def record(self, artifact):
        self.source = artifact.state.function.source_code
        artifact.metadata.file = artifact.name
        artifact.metadata.format = artifact.formatter.name
        self.artifacts[artifact.identity] = artifact.metadata
        self.write()

    def prune(self, archive):
        for identity, artifact in list(self.artifacts.items()):
            formatter = archive.formatter_named(artifact.format)
            path = self.root / f"{artifact.file}{formatter.suffix}"
            if not path.exists():
                del self.artifacts[identity]
        used_values = set()
        for artifact in self.artifacts.values():
            used_values.update(artifact.argument_hashes.values())
            used_values.update(artifact.key.values())
        self.values = {
            value_hash: value
            for value_hash, value in self.values.items()
            if value_hash in used_values
        }
        self.write()

    def write(self):
        data = {
            "version": __version__,
            "qualname": self.qualname,
            "source": self.source,
            "artifacts": {
                identity: vars(artifact)
                for identity, artifact in self.artifacts.items()
            },
            "values": {
                value_hash: vars(value)
                for value_hash, value in self.values.items()
            },
        }
        with _transactional_path(self.root / "meta.json") as temp_path:
            text = json.dumps(data, indent=2, sort_keys=True) + "\n"
            temp_path.write_text(text, encoding="utf-8")


@dataclasses.dataclass
class CacheState:
    function: object
    archive: Archive
    refresh: bool
    read_only: bool
    metadata: StageMetadata

    @property
    def path(self):
        return self.metadata.root

    def arguments_metadata(self, arguments, identity_names=()):
        hash_names = (
            set(arguments) - set(identity_names)
            if self.function.reject_changed
            else set()
        )
        argument_hashes = {}
        audit = {}
        for name, value in arguments.items():
            if name in hash_names:
                argument_hashes[name] = self.hash_value(value)
            else:
                audit[name] = str(value)[:_AUDIT_LIMIT]
        metadata = types.SimpleNamespace(
            arguments=audit,
            argument_hashes=argument_hashes,
            arguments_hash=None,
            key={},
            key_hash=None,
            file=None,
            format=None,
        )
        if self.function.reject_changed:
            text = json.dumps(argument_hashes, sort_keys=True)
            metadata.arguments_hash = _digest(text)
        return metadata

    def hash_value(self, value):
        formatter = self.archive.formatter_for(value)
        value_hash = formatter.hash(value)
        self.metadata.values[value_hash] = types.SimpleNamespace(
            display=str(value)[:_AUDIT_LIMIT], format=formatter.name
        )
        return value_hash

    def artifact(self, name, metadata):
        identity = metadata.key_hash or _digest(self.function.name)
        if self.function.source_code is not None:
            source_hash = _digest(self.function.source_code)
            identity = _digest(identity + source_hash)
            name = f"{name},{source_hash}"
        saved = self.metadata.artifacts.get(identity)
        formatter = saved and self.archive.formatter_named(saved.format)
        name = saved.file if saved else name
        path = self.path / f"{name}{formatter.suffix}" if formatter else None
        return Artifact(
            state=self,
            identity=identity,
            name=name,
            metadata=metadata,
            saved=saved,
            formatter=formatter,
            path=path,
        )


class CacheMismatchError(PypeliteError):
    """A cache hit was produced from different arguments."""


@dataclasses.dataclass
class Artifact:
    state: CacheState
    identity: str
    name: str
    metadata: types.SimpleNamespace
    saved: types.SimpleNamespace | None
    formatter: ArchiveFormat | None
    path: pathlib.Path | None

    def materialize(self, generate):
        if not self.state.refresh:
            hit, value = self.load()
            if hit:
                return value
        value = generate()
        self.write(value)
        return value

    def load(self):
        if self.path is None or not self.path.exists():
            if self.state.read_only:
                raise RuntimeError("read-only pipeline cannot write")
            return False, None
        if (
            self.state.function.reject_changed
            and self.saved.arguments_hash != self.metadata.arguments_hash
        ):
            raise CacheMismatchError(
                f"cached {self.state.function.name!r} called with "
                "different arguments"
            )
        self.state.metadata.touched.add(self.path.name)
        return True, self.formatter.load(self.path)

    def write(self, value):
        self.state.path.mkdir(parents=True, exist_ok=True)
        self.formatter = self.state.archive.formatter_for(value)
        self.path = self.state.path / f"{self.name}{self.formatter.suffix}"
        with _transactional_path(self.path) as temp_path:
            self.formatter.dump(temp_path, value)
        self.state.metadata.touched.add(self.path.name)
        self.state.metadata.record(self)


def _resolve(module_name, name):
    return getattr(importlib.import_module(module_name), name)


@dataclasses.dataclass
class CachedFunction:
    generate: object
    name: str
    qualname: str
    source_code: str | None
    reject_changed: bool
    skip_value: object
    signature: object

    registry: typing.ClassVar[dict] = {}

    def __call__(self, *args, **kwargs):
        if (pipe := _ACTIVE_PIPELINE.get(None)) is None:
            raise RuntimeError("cached function called outside pipeline()")
        return pipe.run(self, args, kwargs)

    def bound(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    def __reduce__(self):
        return _resolve, (self.generate.__module__, self.generate.__name__)


class Checkpoint(CachedFunction):
    def run(self, state, args, kwargs):
        arguments = self.bound(args, kwargs)
        call = functools.partial(self.generate, *args, **kwargs)
        metadata = state.arguments_metadata(arguments)
        artifact = state.artifact("artifact", metadata)
        return artifact.materialize(
            lambda: self.generate_value(state, arguments, call)
        )

    def generate_value(self, state, arguments, call):
        return call()


@dataclasses.dataclass
class Batching:
    batch_size: object = None
    workers: object = None
    pool_type: object = None

    def items(self, value):
        if hasattr(value, "iterrows"):
            return (item for _index, item in value.iterrows())
        return iter(value)

    def batch_sizes(self, item_count):
        while item_count:
            size = self.batch_size or max(
                1, (item_count + 2 * self.workers - 1) // (2 * self.workers)
            )
            size = min(size, item_count)
            yield size
            item_count -= size

    def batches(self, items, item_count):
        item_iter = iter(items)
        for size in self.batch_sizes(item_count):
            yield list(itertools.islice(item_iter, size))

    def worker_map(self, call, arguments, batches):
        with self.pool_type(self.workers) as worker_pool:
            worker = functools.partial(call, arguments)
            return list(worker_pool.map(worker, batches))


@dataclasses.dataclass
class BatchCheckpoint(Batching, Checkpoint):
    batch: object = None

    def generate_value(self, state, arguments, _call):
        items = arguments[self.batch]
        batches = self.batches(self.items(items), len(items))
        values = self.worker_map(self.batch_call, arguments, batches)
        if not values:
            return []
        return state.archive.formatter_for(values[0]).concat(values)

    def batch_call(self, arguments, items):
        return self.generate(**{**arguments, self.batch: items})


@dataclasses.dataclass
class Stage(CachedFunction):
    key: object = None

    def run(self, state, args, kwargs):
        arguments = self.bound(args, kwargs)
        return self.artifact(state, arguments).materialize(
            lambda: self.generate(*args, **kwargs)
        )

    def artifact(self, state, arguments, key_input=None, identity_names=None):
        direct = key_input is None or key_input is arguments
        key_input = arguments if key_input is None else key_input
        key = self.key_value(key_input)
        key_hashes = {
            name: state.hash_value(value) for name, value in key.items()
        }
        if identity_names is None:
            identity_names = arguments if direct and callable(self.key) else key
        metadata = state.arguments_metadata(arguments, identity_names)
        metadata.key = key_hashes
        metadata.key_hash = _digest(json.dumps(key_hashes, sort_keys=True))
        values = state.metadata.values
        parts = (
            values[value_hash].display for value_hash in key_hashes.values()
        )
        readable = re.sub(r"[^A-Za-z0-9_.=-]+", "_", "-".join(parts))
        return state.artifact(
            f"{readable[:20] or 'key'}~{metadata.key_hash[:12]}",
            metadata,
        )

    def key_value(self, item):
        if callable(self.key):
            return {"key": self.key(**item)}
        if self.key is True:
            return item if hasattr(item, "items") else {"key": item}
        if isinstance(self.key, str):
            return {self.key: self.field(item, self.key)}
        return {name: self.field(item, name) for name in self.key}

    def field(self, item, name):
        try:
            return item[name]
        except (KeyError, TypeError):
            return getattr(item, name)


@dataclasses.dataclass
class ItemStage(Batching, Stage):
    argument: object = None
    map_items: bool = False

    def batch_call(self, arguments, items):
        if self.map_items:
            return [
                self.generate(**{**arguments, self.argument: item})
                for item in items
            ]
        return self.generate(**{**arguments, self.argument: items})

    def run(self, state, args, kwargs):
        arguments = self.bound(args, kwargs)
        values = []
        pending = []
        for index, item in enumerate(self.items(arguments[self.argument])):
            item_args = {**arguments, self.argument: item}
            key_input = item_args if self.map_items else item
            identity_names = None if self.map_items else (self.argument,)
            artifact = self.artifact(
                state, item_args, key_input, identity_names
            )
            if not state.refresh:
                hit, value = artifact.load()
                if hit:
                    values.append(value)
                    continue
            values.append(None)
            pending.append((index, item, artifact))
        if pending:
            pending_batches = list(self.batches(pending, len(pending)))
            item_batches = (
                [entry[1] for entry in batch] for batch in pending_batches
            )
            worker_batches = self.worker_map(
                self.batch_call, arguments, item_batches
            )
            new_batches = [list(batch) for batch in worker_batches]
            for pending_batch, new_values in zip(pending_batches, new_batches):
                if len(new_values) != len(pending_batch):
                    raise ValueError(
                        f"batched stage {self.name!r} returned "
                        f"{len(new_values)} outputs for "
                        f"{len(pending_batch)} items"
                    )
                for (index, _item, artifact), value in zip(
                    pending_batch, new_values
                ):
                    values[index] = value
                    artifact.write(value)
        return values


_POOL_TYPES = {
    "thread": concurrent.futures.ThreadPoolExecutor,
    "process": concurrent.futures.ProcessPoolExecutor,
}


def _validate_sizes(batch_size, workers):
    for name, value in (("batch_size", batch_size), ("workers", workers)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise PypeliteError(f"{name} must be a positive int")


def _decorator(name, archive, kind, function_class, **options):
    source = options.pop("source")

    def decorate(generate):
        step_name = str(name or generate.__name__)
        if step_name in {".", ".."} or not re.fullmatch(
            r"[A-Za-z0-9_.=-]+", step_name
        ):
            raise ValueError(f"illegal {kind} filename part: {step_name!r}")
        qualname = f"{generate.__module__}.{generate.__qualname__}"
        try:
            source_code = inspect.getsource(generate) if source else None
        except (OSError, TypeError):
            source_code = None
        function = function_class(
            generate=generate,
            name=step_name,
            qualname=qualname,
            source_code=source_code,
            signature=inspect.signature(generate),
            **options,
        )
        CachedFunction.registry[step_name, qualname] = (function, archive)
        functools.update_wrapper(function, generate)
        return function

    return decorate


def checkpoint(
    *,
    name=None,
    archive="default",
    source=False,
    reject_changed=True,
    batch=None,
    batch_size=None,
    skip_value=None,
    workers=None,
    executor=None,
):
    """Save one result for a pipeline checkpoint.

    Call arguments are recorded but do not select separate artifacts.
    ``batch`` computes one final result in worker-backed batches.
    Cache hits reject changed arguments unless ``reject_changed=False``.
    Explicit ``batch_size`` and ``workers`` values must be positive integers.

    Parameters
    ----------
    name
        Archive directory and run-control name. Defaults to the decorated
        function name.
    archive
        Name of the archive supplied to :func:`pipeline`. Defaults to
        ``"default"``.
    source
        Include source code in artifact identity. Source changes then create
        a new checkpoint artifact.
    reject_changed
        Reject a cache hit when any call argument differs. Defaults to
        ``True``.
    batch
        Name of the collection parameter passed to worker batches. Omit to
        call the function once normally.
    batch_size
        Maximum items per worker batch. ``None`` chooses sizes automatically.
    skip_value
        Value returned when this checkpoint is selected by ``pipeline(skip=)``.
    workers
        Number of workers. ``None`` uses one worker.
    executor
        ``"thread"`` (the default) or ``"process"``. Used only with
        ``batch``.
    """

    _validate_sizes(batch_size, workers)
    function_class = Checkpoint
    options = dict(
        source=source, reject_changed=reject_changed, skip_value=skip_value
    )
    if batch is not None:
        function_class = BatchCheckpoint
        options.update(
            batch=batch,
            batch_size=batch_size,
            workers=workers if workers is not None else 1,
            pool_type=_POOL_TYPES[executor or "thread"],
        )
    return _decorator(
        name=name,
        archive=archive,
        kind="checkpoint",
        function_class=function_class,
        **options,
    )


def stage(
    *,
    name=None,
    archive="default",
    key=True,
    source=False,
    reject_changed=True,
    vectorize=None,
    batch=None,
    batch_size=None,
    skip_value=None,
    workers=None,
    executor=None,
):
    """Cache function results as a pipeline stage.

    All arguments form the key by default. Set ``key`` to a name, tuple, or
    callable to select the cache identity. ``vectorize`` adapts a scalar
    function to collections; ``batch`` passes collections to the function.
    Cache hits reject changed non-key arguments unless
    ``reject_changed=False``.
    Explicit ``batch_size`` and ``workers`` values must be positive integers.

    Parameters
    ----------
    name
        Archive directory and run-control name. Defaults to the decorated
        function name.
    archive
        Name of the archive supplied to :func:`pipeline`. Defaults to
        ``"default"``.
    key
        Artifact identity selector. ``True`` uses all arguments; a string or
        tuple selects fields; a callable returns a derived key.
    source
        Include source code in artifact identity. Source changes then create
        new artifacts.
    reject_changed
        Reject a cache hit when any non-key argument differs from the call
        that produced it. Defaults to ``True``.
    vectorize
        Name of a scalar parameter. Callers pass a collection and pypelite
        calls the function once per missing item.
    batch
        Name of a collection parameter. Pypelite passes missing items to the
        function in batches. Mutually exclusive with ``vectorize``.
    batch_size
        Maximum items per worker batch. ``None`` chooses sizes automatically.
    skip_value
        Value returned when this stage is selected by ``pipeline(skip=)``.
    workers
        Number of workers. ``None`` uses one worker.
    executor
        ``"thread"`` (the default) or ``"process"``. Used with ``vectorize``
        or ``batch``.
    """

    _validate_sizes(batch_size, workers)
    if vectorize is not None and batch is not None:
        raise ValueError("vectorize and batch are mutually exclusive")
    argument = vectorize if vectorize is not None else batch
    function_class = Stage
    options = dict(
        source=source,
        reject_changed=reject_changed,
        skip_value=skip_value,
        key=key,
    )
    if argument is not None:
        function_class = ItemStage
        options.update(
            argument=argument,
            map_items=vectorize is not None,
            batch_size=batch_size,
            workers=workers if workers is not None else 1,
            pool_type=_POOL_TYPES[executor or "thread"],
        )
    return _decorator(
        name=name,
        archive=archive,
        kind="stage",
        function_class=function_class,
        **options,
    )


class _PipelineComplete(Exception):
    pass


@dataclasses.dataclass
class Pipeline:
    clean: set
    skip: set
    states: dict
    until: object = None

    def run(self, stage, args, kwargs):
        state = self.states[stage.name, stage.qualname]
        if stage.name in self.skip:
            value = stage.skip_value
        else:
            state.metadata.touched.add("meta.json")
            value = stage.run(state, args, kwargs)
        if self.until == stage.name:
            raise _PipelineComplete
        return value

    def finish(self):
        for state in {x.path: x for x in self.states.values()}.values():
            if not self.clean.intersection({"all", state.function.name}):
                continue
            if state.metadata.touched:
                for path in state.path.glob("*"):
                    if path.name not in state.metadata.touched:
                        path.unlink()
                state.metadata.prune(state.archive)
            elif state.path.exists():
                shutil.rmtree(state.path)


def argument_parser(**kwargs):
    """Create a parser for :func:`pipeline` run controls.

    Parameters
    ----------
    **kwargs
        Passed to :class:`argparse.ArgumentParser`.

    Returns
    -------
    argparse.ArgumentParser
        Parser providing ``--refresh``, ``--clean``, ``--skip``, ``--until``,
        and ``--read-only``. Stage-name choices come from registered cached
        functions.
    """

    names = sorted(
        {
            function.name
            for function, _archive in CachedFunction.registry.values()
        }
    )
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument("--refresh", nargs="+", choices=["all", *names])
    parser.add_argument("--clean", nargs="+", choices=["all", *names])
    parser.add_argument("--skip", nargs="+", choices=names)
    parser.add_argument("--until", choices=names)
    parser.add_argument("--read-only", action="store_true")
    return parser


@contextlib.contextmanager
def pipeline(
    path=None,
    /,
    *,
    archive=None,
    archives=None,
    refresh=None,
    clean=None,
    skip=None,
    until=None,
    read_only=False,
):
    """Run checkpoints and cached stages against an archive.

    The context does not plan a DAG. Your Python code decides which stages run.

    Parameters
    ----------
    path
        Default archive directory. Pass it positionally. Ignored when
        ``archive`` is supplied.
    archive
        Preconfigured default :class:`Archive`, including custom formatters.
    archives
        Additional named archives for cached functions that set
        ``archive=...``. Values may be paths or :class:`Archive` instances.
        This mapping cannot contain ``"default"``. Formatter resolution uses
        the named archive, the default archive, then pickle.
    refresh
        List or tuple of cached-function names to rerun and write back. Use
        ``"all"`` for every available function.
    clean
        List or tuple of names to prune after a completed run. Untouched
        artifacts are removed. Use ``"all"`` for every available function.
    skip
        List or tuple of names whose calls return their decorator
        ``skip_value`` without cache access or execution.
    until
        Cached-function name after which the context exits normally.
    read_only
        Use shared archive locks and reject cache misses. Cannot be combined
        with ``refresh`` or ``clean``.
    """

    if _ACTIVE_PIPELINE.get(None) is not None:
        raise RuntimeError("pypelite.pipeline contexts cannot be nested")
    if not all(
        x is None or isinstance(x, (list, tuple))
        for x in (refresh, clean, skip)
    ):
        raise TypeError("pipeline stages must be a list or tuple")
    if read_only and (refresh or clean):
        raise ValueError("read-only pipeline cannot refresh or clean")
    archives = dict(archives or {})
    if "default" in archives:
        raise ValueError("archives contains only additional named archives")
    available = {
        function.name
        for function, archive_name in CachedFunction.registry.values()
        if archive_name in {"default", *archives}
    }
    for control, selected, allowed in (
        ("refresh", refresh, available | {"all"}),
        ("clean", clean, available | {"all"}),
        ("skip", skip, available),
        ("until", [until] if until is not None else None, available),
    ):
        unknown = set(selected or ()) - allowed
        if unknown:
            raise ValueError(
                f"unknown {control}: " + ", ".join(sorted(unknown))
            )
    default = archive or Archive(path)
    fallback_formatters = [*default.formatters, ArchiveFormat()]
    archive_map = {
        "default": Archive(default.path, formatters=fallback_formatters)
    }
    for name, value in archives.items():
        if not isinstance(value, Archive):
            value = Archive(value)
        archive_map[name] = Archive(
            value.path,
            formatters=[*value.formatters, *fallback_formatters],
        )
    with contextlib.ExitStack() as locks:
        locked_archives = {
            value.path.resolve(): value for value in archive_map.values()
        }
        for root in sorted(locked_archives):
            locks.enter_context(locked_archives[root].lock(read_only))
        refresh = set(refresh or ())
        states = {}
        metadata = {}
        for registry_key, registered in CachedFunction.registry.items():
            function, archive_name = registered
            if archive_name not in archive_map:
                continue
            stage_archive = archive_map[archive_name]
            root = stage_archive.path.resolve() / function.name
            if root not in metadata:
                metadata[root] = StageMetadata.load(root, function)
            else:
                metadata[root].check(function)
            states[registry_key] = CacheState(
                function=function,
                archive=stage_archive,
                refresh="all" in refresh or function.name in refresh,
                read_only=read_only,
                metadata=metadata[root],
            )
        pipe = Pipeline(
            clean=set(clean or ()),
            skip=set(skip or ()),
            states=states,
            until=until,
        )
        token = _ACTIVE_PIPELINE.set(pipe)
        try:
            try:
                yield pipe
            except _PipelineComplete:
                pass
            pipe.finish()
        finally:
            _ACTIVE_PIPELINE.reset(token)
