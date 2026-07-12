"""Optional archive formats."""

import importlib.util

import pypelite


def default_formatters():
    """Return formatters for installed optional packages.

    Missing packages are skipped.

    Returns
    -------
    list[pypelite.ArchiveFormat]
        Available pandas, NumPy, and XGBoost formatters in lookup order.
    """

    formatters = []
    if importlib.util.find_spec("pandas") and importlib.util.find_spec(
        "pyarrow"
    ):
        formatters.append(pandas_arrow())
    if importlib.util.find_spec("numpy"):
        formatters.append(numpy_npz())
    if importlib.util.find_spec("xgboost"):
        formatters.append(xgboost_ubj())
    return formatters


def archive(path, formatters=None, defaults=True):
    """Create an archive with optional formatters.

    Explicit formatters replace defaults for the same value type.

    Parameters
    ----------
    path
        Archive directory.
    formatters
        Explicit formatter instances. A formatter replaces an installed
        default with the same ``value_type``.
    defaults
        Include compatible optional formatters for installed packages.

    Returns
    -------
    pypelite.Archive
        Configured archive. Pipeline-level pickle fallback is added later.
    """

    archive_formatters = default_formatters() if defaults else []
    override_types = {formatter.value_type for formatter in formatters or []}
    archive_formatters = [
        formatter
        for formatter in archive_formatters
        if formatter.value_type not in override_types
    ]
    archive_formatters.extend(formatters or [])
    return pypelite.Archive(path, formatters=archive_formatters)


class PandasArrowFormat(pypelite.ArchiveFormat):
    """Feather DataFrame format."""

    name = "pandas.arrow"
    suffix = ".feather"

    def load(self, path):
        import pyarrow
        import pyarrow.ipc

        with pyarrow.memory_map(str(path), "r") as source:
            return pyarrow.ipc.open_file(source).read_all().to_pandas()

    def dump(self, path, value):
        import pyarrow
        import pyarrow.ipc

        table = pyarrow.Table.from_pandas(value)
        with pyarrow.OSFile(str(path), "wb") as sink:
            with pyarrow.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def concat(self, values):
        import pandas

        return pandas.concat(values, ignore_index=True)


class PandasCsvFormat(PandasArrowFormat):
    """CSV DataFrame format."""

    name = "pandas.csv"
    suffix = ".csv"

    def load(self, path):
        import pandas

        return pandas.read_csv(path)

    def dump(self, path, value):
        value.to_csv(path, index=False)


class NumpyNpzFormat(pypelite.ArchiveFormat):
    """NPZ array format."""

    name = "numpy.npz"
    suffix = ".npz"

    def __init__(self, compressed=True):
        self.compressed = compressed

    def load(self, path):
        import numpy

        with numpy.load(path, allow_pickle=False) as data:
            return data["value"]

    def dump(self, path, value):
        import numpy

        save = numpy.savez_compressed if self.compressed else numpy.savez
        save(path, value=value)

    def concat(self, values):
        import numpy

        return numpy.concatenate(values)


class XGBoostUbjFormat(pypelite.ArchiveFormat):
    """XGBoost UBJ model format."""

    name = "xgboost.ubj"
    suffix = ".ubj"

    def load(self, path):
        import xgboost

        model = xgboost.Booster()
        model.load_model(path)
        return model

    def dump(self, path, value):
        value.save_model(path)


class TensorFlowKerasWeightsFormat(pypelite.ArchiveFormat):
    """TensorFlow Keras weights in HDF5."""

    name = "tensorflow.keras.weights"
    suffix = ".weights.h5"

    def __init__(self, model_factory):
        self.model_factory = model_factory

    def load(self, path):
        import h5py

        model = self.model_factory()
        with h5py.File(path) as weights:
            model.set_weights(
                [weights[str(index)][...] for index in range(len(weights))]
            )
        return model

    def dump(self, path, value):
        import h5py

        with h5py.File(path, "w") as weights:
            for index, variable in enumerate(value.weights):
                weights.create_dataset(str(index), data=variable.numpy())


def pandas_arrow():
    """Use Feather for DataFrames.

    This is the default pandas choice.

    Returns
    -------
    pypelite.ArchiveFormat
        Feather formatter for ``pandas.DataFrame`` values.
    """

    import pandas

    formatter = PandasArrowFormat()
    formatter.value_type = pandas.DataFrame
    return formatter


def pandas_csv():
    """Use CSV for DataFrames.

    CSV is mainly useful when cached tables need to be opened with text tools.

    Returns
    -------
    pypelite.ArchiveFormat
        CSV formatter for ``pandas.DataFrame`` values.
    """

    import pandas

    formatter = PandasCsvFormat()
    formatter.value_type = pandas.DataFrame
    return formatter


def numpy_npz(compressed=True):
    """Use NPZ for NumPy arrays.

    Turn compression off when speed matters more than archive size.

    Parameters
    ----------
    compressed
        Use ``numpy.savez_compressed`` instead of ``numpy.savez``.

    Returns
    -------
    pypelite.ArchiveFormat
        Formatter for ``numpy.ndarray`` values.
    """

    import numpy

    formatter = NumpyNpzFormat(compressed=compressed)
    formatter.value_type = numpy.ndarray
    return formatter


def xgboost_ubj():
    """Use XGBoost's UBJ model format for boosters.

    Returns
    -------
    pypelite.ArchiveFormat
        UBJ formatter for ``xgboost.Booster`` values.
    """

    import xgboost

    formatter = XGBoostUbjFormat()
    formatter.value_type = xgboost.Booster
    return formatter


def keras_weights(model_type, model_factory):
    """Use Keras weights files for a concrete model type.

    ``model_factory`` should rebuild the model before weights are loaded.

    Parameters
    ----------
    model_type
        Concrete model class handled by the formatter.
    model_factory
        Zero-argument callable returning the initialized model structure used
        when loading weights.

    Returns
    -------
    pypelite.ArchiveFormat
        HDF5 weights formatter for ``model_type``.
    """

    formatter = TensorFlowKerasWeightsFormat(model_factory)
    formatter.value_type = model_type
    return formatter
