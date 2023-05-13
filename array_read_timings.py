import perfplot
import pickle
import numpy
import h5py
import tables
import zarr


def setup(n):
    data = numpy.random.rand(n,n)
    # write all files
    #
    numpy.save("out.npy", data)
    #
    f = h5py.File("out.h5", "w")
    f.create_dataset("data", data=data)
    f.close()
    #
    with open("test.pkl", "wb") as f:
        pickle.dump(data, f)
    #
    f = tables.open_file("pytables.h5", mode="w")
    gcolumns = f.create_group(f.root, "columns", "data")
    f.create_array(gcolumns, "data", data, "data")
    f.close()
    #
    zarr.save("out.zip", data)


def npy_read(data):
    return numpy.load("out.npy")


def hdf5_read(data):
    f = h5py.File("out.h5", "r")
    out = f["data"][()]
    f.close()
    return out


def pickle_read(data):
    with open("test.pkl", "rb") as f:
        out = pickle.load(f)
    return out


def pytables_read(data):
    f = tables.open_file("pytables.h5", mode="r")
    out = f.root.columns.data[()]
    f.close()
    return out


def zarr_read(data):
    return zarr.load("out.zip")


b = perfplot.bench(
    setup=setup,
    kernels=[
        npy_read,
        hdf5_read,
        pickle_read,
        pytables_read,
        zarr_read,
    ],
    n_range=[2 ** k for k in range(14)],
    xlabel="Side Length of Square Array"
)
b.save("out2.png")
b.show()