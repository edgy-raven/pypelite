import importlib.util
import json
import multiprocessing

import pytest

import pypelite
import pypelite.configs


@pypelite.checkpoint(
    batch="records",
    batch_size=2,
    workers=2,
    executor="process",
)
def process_judge(records):
    return [record["case_id"].upper() for record in records]


@pypelite.stage(
    key="case_id",
    batch="records",
    workers=2,
    executor="process",
)
def process_score(records):
    return [record["case_id"].upper() for record in records]


def hold_pipeline(path, ready, entered, release, read_only=False):
    ready.set()
    with pypelite.pipeline(path, read_only=read_only):
        entered.set()
        release.wait()


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def iterrows(self):
        return enumerate(self.rows)


_BASE_REGISTRY = dict(pypelite.CachedFunction.registry)


@pytest.fixture(autouse=True)
def restore_registry():
    registry = pypelite.CachedFunction.registry
    registry.clear()
    registry.update(_BASE_REGISTRY)
    yield
    registry.clear()
    registry.update(_BASE_REGISTRY)


def test_public_api_contract(tmp_path):
    assert pypelite.__version__ == "0.1.12"
    assert issubclass(pypelite.CacheMismatchError, pypelite.PypeliteError)

    for decorator in (pypelite.checkpoint, pypelite.stage):
        for option in ("batch_size", "workers"):
            for value in (0, -1, 1.5, True, "1"):
                with pytest.raises(
                    pypelite.PypeliteError,
                    match=f"{option} must be a positive int",
                ):
                    decorator(batch="items", **{option: value})

    pypelite.checkpoint(batch="items", batch_size=None, workers=None)
    pypelite.stage(batch="items", batch_size=1, workers=1)

    with pytest.raises(ValueError, match="additional named archives"):
        with pypelite.pipeline(tmp_path, archives={"default": tmp_path}):
            pass

    with pytest.raises(TypeError, match="list or tuple"):
        with pypelite.pipeline(tmp_path, refresh="stage"):
            pass

    @pypelite.checkpoint()
    def load_prices(symbols):
        return symbols

    @pypelite.checkpoint()
    def build_features(prices):
        return [price * 2 for price in prices]

    @pypelite.checkpoint()
    def train_model(features):
        return sum(features)

    with pypelite.pipeline(tmp_path / "flagship"):
        prices = load_prices([1, 2, 3])
        assert train_model(build_features(prices)) == 12

    parser = pypelite.argument_parser(add_help=False)
    args = parser.parse_args(["--refresh", "all", "load_prices", "--read-only"])
    assert args.refresh == ["all", "load_prices"]
    assert args.read_only
    with pytest.raises(SystemExit):
        parser.parse_args(["--clean", "bad"])

    path = tmp_path / "refresh"
    with pytest.raises(ValueError, match="unknown refresh: bad"):
        with pypelite.pipeline(path, refresh=["bad"]):
            pass
    assert not path.exists()

    path = tmp_path / "until"
    with pytest.raises(ValueError, match="unknown until: bad"):
        with pypelite.pipeline(path, until="bad"):
            pass
    assert not path.exists()


def test_archive_formatter_precedence(tmp_path):
    class TextFormat(pypelite.ArchiveFormat):
        value_type = str

        def load(self, path):
            return path.read_text(encoding="utf-8")

        def dump(self, path, value):
            path.write_text(value, encoding="utf-8")

    class DefaultTextFormat(TextFormat):
        name = "default-text"
        suffix = ".default"

    class CurrentTextFormat(TextFormat):
        name = "current-text"
        suffix = ".current"

    base_fmt = DefaultTextFormat()
    named_fmt = CurrentTextFormat()
    base = pypelite.Archive(tmp_path / "default", formatters=[base_fmt])
    named = pypelite.Archive(tmp_path / "current", formatters=[named_fmt])
    plain = pypelite.Archive(tmp_path / "pickle")

    @pypelite.checkpoint(archive="current")
    def current_value():
        return "current"

    @pypelite.checkpoint(archive="inherited")
    def inherited_value():
        return "inherited"

    @pypelite.checkpoint(archive="pickle")
    def pickle_value():
        return 3

    archives = {
        "current": named,
        "inherited": tmp_path / "inherited",
        "pickle": plain,
    }
    with pypelite.pipeline(archive=base, archives=archives):
        assert current_value() == "current"
        assert inherited_value() == "inherited"
        assert pickle_value() == 3
    assert list((tmp_path / "current" / "current_value").glob("*.current"))
    assert list((tmp_path / "inherited" / "inherited_value").glob("*.default"))
    assert list((tmp_path / "pickle" / "pickle_value").glob("*.pkl"))
    assert base.formatters == [base_fmt]
    assert named.formatters == [named_fmt]
    assert plain.formatters == []


@pytest.mark.parametrize("read_only", [False, True])
def test_pipeline_locks_archive_across_processes(tmp_path, read_only):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    entered = context.Event()
    release = context.Event()
    process = context.Process(
        target=hold_pipeline,
        args=(tmp_path, ready, entered, release, read_only),
    )
    try:
        with pypelite.pipeline(tmp_path, read_only=read_only):
            process.start()
            assert ready.wait(5)
            assert entered.wait(5 if read_only else 0.2) is read_only
        if not read_only:
            assert entered.wait(5)
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_pipeline_run_controls_and_archives_document_main_flow(tmp_path):
    version = {"value": 1}
    calls = []
    run_archive = tmp_path / "run"
    shared_archive = pypelite.Archive(tmp_path / "shared")

    @pypelite.checkpoint(reject_changed=False)
    def load_records(path):
        return f"{path}:v{version['value']}"

    @pypelite.checkpoint(skip_value=None)
    def history(case_id):
        return f"history {case_id}"

    @pypelite.stage(key="case_id", reject_changed=False)
    def score(case_id, run_id, value):
        return f"{case_id}:{run_id}:{value}:v{version['value']}"

    @pypelite.stage(key="case_id", batch="records", workers=2)
    def batch_score(records):
        calls.append([record["case_id"] for record in records])
        return [
            f"{record['case_id']}:v{version['value']}" for record in records
        ]

    @pypelite.checkpoint(
        batch="records", batch_size=2, workers=2, reject_changed=False
    )
    def judge(records):
        calls.append([record["case_id"] for record in records])
        return [
            f"{record['case_id']}:v{version['value']}" for record in records
        ]

    @pypelite.checkpoint(archive="shared")
    def prices(symbol):
        return f"{symbol}:v{version['value']}"

    @pypelite.checkpoint()
    def missing():
        raise AssertionError("read-only miss ran the function")

    with pypelite.pipeline(run_archive, archives={"shared": shared_archive}):
        assert load_records("a.json") == "a.json:v1"
        assert history("a") == "history a"
        assert score("a", "old", 3) == "a:old:3:v1"
        assert batch_score(
            [
                {"case_id": "a", "payload": "old"},
                {"case_id": "b", "payload": "old"},
            ]
        ) == [
            "a:v1",
            "b:v1",
        ]
        assert judge(
            [
                {"case_id": "a"},
                {"case_id": "b"},
                {"case_id": "c"},
            ]
        ) == ["a:v1", "b:v1", "c:v1"]
        assert prices("ABC") == "ABC:v1"

    version["value"] = 2

    with pypelite.pipeline(
        run_archive,
        archives={"shared": shared_archive},
        skip=["history"],
        clean=["batch_score"],
    ):
        assert load_records("b.json") == "a.json:v1"
        assert history("a") is None
        assert score("a", "new", 99) == "a:old:3:v1"
        assert batch_score(
            [
                {"case_id": "a", "payload": "new"},
                {"case_id": "c", "payload": "new"},
            ]
        ) == [
            "a:v1",
            "c:v2",
        ]
        assert judge([{"case_id": "z"}]) == ["a:v1", "b:v1", "c:v1"]
        assert prices("ABC") == "ABC:v1"

    batch_metadata = json.loads(
        (run_archive / "batch_score" / "meta.json").read_text()
    )
    assert {
        batch_metadata["values"][artifact["key"]["case_id"]]["display"]
        for artifact in batch_metadata["artifacts"].values()
    } == {"a", "c"}
    assert {
        value["display"] for value in batch_metadata["values"].values()
    } == {"a", "c"}

    with pypelite.pipeline(
        run_archive,
        archives={"shared": shared_archive},
        refresh=["load_records"],
    ):
        assert load_records("b.json") == "b.json:v2"
        assert score("b", "run", 4) == "b:run:4:v2"

    with pypelite.pipeline(
        run_archive, archives={"shared": shared_archive}, read_only=True
    ):
        assert load_records("ignored.json") == "b.json:v2"
        with pytest.raises(RuntimeError, match="read-only"):
            score("missing", "run", 4)
        with pytest.raises(RuntimeError, match="read-only"):
            missing()

    stopped = []
    with pypelite.pipeline(run_archive, until="load_records"):
        stopped.append("before")
        load_records("c.json")
        stopped.append("not reached")
    assert stopped == ["before"]

    @pypelite.checkpoint()
    def unused():
        return "value"

    with pypelite.pipeline(run_archive):
        assert unused() == "value"
    with pypelite.pipeline(run_archive, clean=["unused"]):
        pass
    assert not (run_archive / "unused").exists()


def test_vectorized_batched_and_process_execution(tmp_path):
    calls = []
    scalar_calls = []

    @pypelite.stage(vectorize="case", workers=2)
    def scalar_score(case):
        calls.append([case["case_id"]])
        return case["case_id"].upper()

    @pypelite.stage(key="case_id", batch="cases", workers=2)
    def frame_score(cases):
        calls.append([case["case_id"] for case in cases])
        return [case["case_id"].upper() for case in cases]

    @pypelite.stage(key="case_id", batch="cases", workers=2)
    def nullable_score(cases):
        return [
            None if case["case_id"] == "b" else case["case_id"].upper()
            for case in cases
        ]

    @pypelite.stage(batch="values", workers=2)
    def double(values):
        scalar_calls.extend(values)
        return [value * 2 for value in values]

    @pypelite.stage(key="case_id", batch="records")
    def wrong_size(records):
        return []

    with pypelite.pipeline(tmp_path):
        assert scalar_score([{"case_id": "x"}, {"case_id": "y"}]) == [
            "X",
            "Y",
        ]
        assert frame_score(FakeFrame([{"case_id": "a"}])) == ["A"]
        assert frame_score({"case_id": name} for name in ["b", "c"]) == [
            "B",
            "C",
        ]
        assert nullable_score([{"case_id": "a"}, {"case_id": "b"}]) == [
            "A",
            None,
        ]
        assert double([1, 2]) == [2, 4]
        assert double([2, 3]) == [4, 6]
        assert process_judge(
            [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
        ) == ["A", "B", "C"]
        assert process_score([{"case_id": "a"}, {"case_id": "b"}]) == [
            "A",
            "B",
        ]
    assert sorted(calls) == [["a"], ["b"], ["c"], ["x"], ["y"]]
    assert sorted(scalar_calls) == [1, 2, 3]
    with pytest.raises(ValueError, match="returned 0 outputs for 1 items"):
        with pypelite.pipeline(tmp_path / "invalid"):
            wrong_size([{"case_id": "a"}])
    assert not list((tmp_path / "invalid" / "wrong_size").glob("*.pkl"))


def test_cache_identity_metadata_and_argument_rejection(tmp_path):
    version = {"value": 1}

    notebook = {}
    exec(
        compile(
            "def dynamic(value):\n    return value\n",
            "<ipython-input-1-test>",
            "exec",
        ),
        notebook,
    )
    dynamic = pypelite.stage(source=True)(notebook["dynamic"])

    @pypelite.stage(source=True)
    def automatic(name):
        return f"{name}:v{version['value']}"

    @pypelite.stage(key="name", source=True, reject_changed=False)
    def source_key(name, payload):
        return f"{name}:{payload}:v{version['value']}"

    @pypelite.checkpoint()
    def checked_checkpoint(value):
        return value

    @pypelite.stage(key="name")
    def checked_stage(name, payload):
        return payload

    with pypelite.pipeline(tmp_path):
        assert dynamic("value") == "value"
        assert automatic("a") == "a:v1"
        assert source_key("a", "old") == "a:old:v1"
        assert checked_checkpoint("old") == "old"
        assert checked_stage("item", "old") == "old"

    version["value"] = 2

    with pypelite.pipeline(tmp_path):
        assert automatic("b") == "b:v2"
        assert automatic("a") == "a:v1"
        assert source_key("a", "new") == "a:old:v1"
        with pytest.raises(pypelite.CacheMismatchError):
            checked_checkpoint("new")
        with pytest.raises(pypelite.CacheMismatchError):
            checked_stage("item", "new")

    metadata = json.loads(
        (tmp_path / "automatic" / "meta.json").read_text(encoding="utf-8")
    )
    assert metadata["version"] == pypelite.__version__
    assert metadata["qualname"].endswith(
        "test_cache_identity_metadata_and_argument_rejection.<locals>.automatic"
    )
    assert "def automatic" in metadata["source"]
    artifacts = list(metadata["artifacts"].values())
    assert {artifact["file"].split("~", 1)[0] for artifact in artifacts} == {
        "a",
        "b",
    }
    name_hashes = [artifact["key"]["name"] for artifact in artifacts]
    assert sorted(
        metadata["values"][value_hash]["display"] for value_hash in name_hashes
    ) == ["a", "b"]
    assert all(
        set(metadata["values"][value_hash]) == {"display", "format"}
        for value_hash in name_hashes
    )
    assert all(
        set(artifact)
        == {
            "arguments",
            "argument_hashes",
            "arguments_hash",
            "key",
            "key_hash",
            "file",
            "format",
        }
        for artifact in artifacts
    )
    dynamic_metadata = json.loads(
        (tmp_path / "dynamic" / "meta.json").read_text(encoding="utf-8")
    )
    assert dynamic_metadata["source"] is None

    for invalid in (
        {**metadata, "version": "0.1.11"},
        {**metadata, "extra": "invalid"},
    ):
        (tmp_path / "automatic" / "meta.json").write_text(json.dumps(invalid))
        with pytest.raises(
            pypelite.PypeliteError, match="invalid metadata format"
        ):
            with pypelite.pipeline(tmp_path):
                pass

    (tmp_path / "automatic" / "meta.json").write_text("{")
    with pytest.raises(pypelite.PypeliteError, match="invalid metadata format"):
        with pypelite.pipeline(tmp_path):
            pass


def test_optional_formatters_round_trip(tmp_path):
    formatter_names = {
        formatter.name for formatter in pypelite.configs.default_formatters()
    }
    if importlib.util.find_spec("pyarrow"):
        assert "pandas.arrow" in formatter_names
    else:
        assert "pandas.arrow" not in formatter_names

    numpy = pytest.importorskip("numpy")
    npz = pypelite.configs.numpy_npz()
    array = numpy.array([[1, 2], [3, 4]])
    npz.dump(tmp_path / "array.npz", array)
    numpy.testing.assert_array_equal(npz.load(tmp_path / "array.npz"), array)

    calls = []

    @pypelite.stage()
    def array_argument(value):
        calls.append(value)
        return value.sum()

    argument_archive = pypelite.configs.archive(
        tmp_path / "arguments", formatters=[npz], defaults=False
    )
    with pypelite.pipeline(archive=argument_archive):
        assert array_argument(array) == 10
    with pypelite.pipeline(archive=argument_archive):
        assert array_argument(array.copy()) == 10
    assert len(calls) == 1
    argument_metadata = json.loads(
        (tmp_path / "arguments" / "array_argument" / "meta.json").read_text()
    )
    artifact = next(iter(argument_metadata["artifacts"].values()))
    value_hash = artifact["key"]["value"]
    assert argument_metadata["values"][value_hash]["format"] == "numpy.npz"

    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({"symbol": ["ABC", "XYZ"], "price": [1.5, 2.0]})
    csv = pypelite.configs.pandas_csv()
    csv.dump(tmp_path / "frame.csv", frame)
    pandas.testing.assert_frame_equal(csv.load(tmp_path / "frame.csv"), frame)

    pytest.importorskip("pyarrow")
    arrow = pypelite.configs.pandas_arrow()
    arrow.dump(tmp_path / "frame.feather", frame)
    pandas.testing.assert_frame_equal(
        arrow.load(tmp_path / "frame.feather"), frame
    )

    tensorflow = pytest.importorskip("tensorflow")

    def model_factory():
        model = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.Input(shape=(2,)),
                tensorflow.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        return model

    model = model_factory()
    model.set_weights(
        [
            numpy.array([[1.0], [2.0]]),
            numpy.array([3.0]),
        ]
    )
    weights = pypelite.configs.keras_weights(type(model), model_factory)
    weights.dump(tmp_path / "model.weights.h5", model)
    restored = weights.load(tmp_path / "model.weights.h5")
    for original, loaded in zip(model.get_weights(), restored.get_weights()):
        assert (original == loaded).all()

    xgboost = pytest.importorskip("xgboost")
    train = xgboost.DMatrix(
        numpy.array([[0.0, 1.0], [1.0, 0.0]]),
        label=numpy.array([0.0, 1.0]),
    )
    booster = xgboost.train(
        {"objective": "reg:squarederror"}, train, num_boost_round=1
    )
    ubj = pypelite.configs.xgboost_ubj()
    ubj.dump(tmp_path / "model.ubj", booster)
    numpy.testing.assert_allclose(
        ubj.load(tmp_path / "model.ubj").predict(train),
        booster.predict(train),
    )


def test_stage_identity_and_name_collisions(tmp_path):
    unordered_calls = []
    first = {
        "a": 1,
        "nested": [{"x": 2, "y": 3}],
        "members": {"a", "b"},
    }
    second = {
        "members": {"b", "a"},
        "nested": [{"y": 3, "x": 2}],
        "a": 1,
    }

    def bad_stage():
        return "bad"

    with pytest.raises(ValueError, match="illegal stage filename"):
        pypelite.stage(name="bad/name")(bad_stage)

    @pypelite.stage(name="failure_filename", key="case_id")
    def filename_case(case_id):
        return case_id

    @pypelite.stage(key=("left", "right"))
    def tuple_case(left, right):
        return f"{left}:{right}"

    @pypelite.stage()
    def unordered(value):
        unordered_calls.append(value)
        return len(unordered_calls)

    @pypelite.stage(key=lambda record: record["case_id"])
    def derived_key(record):
        return record["payload"]

    @pypelite.checkpoint(name="shared", archive="first")
    def first_stage():
        return "first"

    @pypelite.checkpoint(name="shared", archive="second")
    def second_stage():
        return "second"

    with pypelite.pipeline(tmp_path):
        assert filename_case("a/b") == "a/b"
        assert filename_case("a" * 40) == "a" * 40
        assert tuple_case("a", "b_c") == "a:b_c"
        assert tuple_case("a_b", "c") == "a_b:c"
        assert derived_key({"case_id": "a", "payload": "old"}) == "old"
        assert derived_key({"case_id": "a", "payload": "new"}) == "old"
        assert unordered(first) == 1
        assert unordered(second) == 1
        assert unordered({"a", "b"}) == 2
        assert unordered(frozenset({"b", "a"})) == 2
    assert unordered_calls == [first, {"a", "b"}]
    filenames = list((tmp_path / "failure_filename").glob("*.pkl"))
    assert any(path.name.startswith("a_b~") for path in filenames)
    assert all(len(path.stem.split("~", 1)[0]) <= 20 for path in filenames)
    with pypelite.pipeline(
        tmp_path / "unused",
        archives={
            "first": tmp_path / "first",
            "second": tmp_path / "second",
        },
    ):
        assert first_stage() == "first"
        assert second_stage() == "second"
    with pytest.raises(ValueError, match="stage name 'shared' belongs to"):
        with pypelite.pipeline(
            tmp_path / "unused",
            archives={"second": tmp_path / "first"},
        ):
            pass


def test_stages_hash_only_key_arguments(tmp_path):
    class AuditOnly:
        def __str__(self):
            return "x" * 1200

        def __reduce__(self):
            raise AssertionError("non-key argument was serialized")

    class CountingFormat(pypelite.ArchiveFormat):
        name = "counting"

        def __init__(self):
            self.hashed = []

        def hash(self, value):
            self.hashed.append(value)
            return super().hash(value)

    formatter = CountingFormat()
    archive = pypelite.Archive(tmp_path, formatters=[formatter])

    @pypelite.stage(key="case_id", reject_changed=False)
    def keyed(case_id, payload):
        return "result"

    with pypelite.pipeline(archive=archive):
        assert keyed("a", AuditOnly()) == "result"
    assert formatter.hashed == ["a"]
    metadata = json.loads(
        (tmp_path / "keyed" / "meta.json").read_text(encoding="utf-8")
    )
    artifact = next(iter(metadata["artifacts"].values()))
    assert artifact["arguments"]["payload"] == "x" * 1000
    assert artifact["argument_hashes"] == {}
    assert set(metadata["values"]) == set(artifact["key"].values())

    formatter.hashed.clear()

    @pypelite.stage(key="case_id")
    def checked(case_id, payload):
        return payload

    with pypelite.pipeline(archive=archive):
        assert checked("a", "checked") == "checked"
    assert formatter.hashed == ["a", "checked"]
    checked_metadata = json.loads(
        (tmp_path / "checked" / "meta.json").read_text(encoding="utf-8")
    )
    checked_artifact = next(iter(checked_metadata["artifacts"].values()))
    assert set(checked_artifact["key"]) == {"case_id"}
    assert set(checked_artifact["argument_hashes"]) == {"payload"}


def test_artifact_write_is_transactional(tmp_path):
    class FailingTextFormat(pypelite.ArchiveFormat):
        name = "text"
        value_type = str
        suffix = ".txt"

        def __init__(self):
            self.fail = False

        def load(self, path):
            return path.read_text(encoding="utf-8")

        def dump(self, path, value):
            if self.fail:
                path.write_text("partial", encoding="utf-8")
                raise RuntimeError("write failed")
            path.write_text(value, encoding="utf-8")

    formatter = FailingTextFormat()
    archive = pypelite.Archive(tmp_path, formatters=[formatter])
    current = {"value": "old"}

    @pypelite.checkpoint()
    def transactional_text():
        return current["value"]

    with pypelite.pipeline(archive=archive):
        assert transactional_text() == "old"

    artifact_path = tmp_path / "transactional_text" / "artifact.txt"
    meta_path = tmp_path / "transactional_text" / "meta.json"
    old_metadata = meta_path.read_text(encoding="utf-8")
    formatter.fail = True
    current["value"] = "new"

    with pytest.raises(RuntimeError, match="write failed"):
        with pypelite.pipeline(archive=archive, refresh=["transactional_text"]):
            transactional_text()

    assert artifact_path.read_text(encoding="utf-8") == "old"
    assert meta_path.read_text(encoding="utf-8") == old_metadata
    assert not list(artifact_path.parent.glob(".artifact.txt.*"))
