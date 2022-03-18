import gcsfs
import zarr

fs = gcsfs.GCSFileSystem(project='soe-octopi',token='whole-slide-20220214-keys.json')
store = fs.get_mapper('gs://octopi-malaria-whole-slide/zarr-test/test2')
root = zarr.group(store=store)
foo = root.create_group('foo')
bar = foo.create_group('bar')
z1 = bar.zeros('baz', shape=(3000, 3000), chunks=(1000, 1000), dtype='i4')

# pips install fs-gcsfs
# map = fsspec.get_mapper("gcs://...")