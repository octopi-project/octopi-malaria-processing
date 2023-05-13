from sys import version_info

import matplotlib.pyplot as plt

import perfplot
import pickle
import netCDF4
import numpy as np
import h5py
import tables
import zarr


def write_numpy(data):
    np.save("out.npy", data)


def write_hdf5(data):
    with h5py.File("out.h5", "w") as f:
        f.create_dataset("data", data=data)


def write_netcdf(data):
    with netCDF4.Dataset("out.nc", "w") as nc:
        nc.createDimension("len_data", len(data))
        ncdata = nc.createVariable(
            "mydata",
            "float64",
            ("len_data",),
        )
        ncdata[:] = data


def write_pickle(data):
    with open("out.pkl", "wb") as f:
        pickle.dump(data, f)


def write_pytables(data):
    with tables.open_file("out-pytables.h5", mode="w") as f:
        gcolumns = f.create_group(f.root, "columns", "data")
        f.create_array(gcolumns, "data", data, "data")


def write_zarr_zarr(data):
    zarr.save_array("out.zarr", data)


def write_zarr_zip(data):
    zarr.save_array("out.zip", data)


def setup(n):
    data = np.random.rand(n)
    return data


b = perfplot.bench(
    setup=setup,
    kernels=[
        write_numpy,
        write_hdf5,
        write_netcdf,
        write_pickle,
        write_pytables,
        write_zarr_zarr,
        write_zarr_zip,
    ],
    n_range=[2**k for k in range(20)],
    title="Write Comparison",
    xlabel="Side Length of Square Array",
    equality_check=None,
)

plt.text(
    0.0,
    -0.3,
    ", ".join(
        [
            f"Python {version_info.major}.{version_info.minor}.{version_info.micro}",
            f"h5py {h5py.__version__}",
            f"netCDF4 {netCDF4.__version__}",
            f"NumPy {np.__version__}",
            f"PyTables {tables.__version__}",
            f"Zarr {zarr.__version__}",
        ]
    ),
    transform=plt.gca().transAxes,
    fontsize="x-small",
    verticalalignment="top",
)

b.save("out-write.png")
b.show()
