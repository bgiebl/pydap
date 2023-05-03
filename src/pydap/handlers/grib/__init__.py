"""pydap handler for GRIB files."""

import os
import re
import time
from stat import ST_MTIME
from email.utils import formatdate
import numpy as np

from pkg_resources import get_distribution

from ...model import DatasetType, GridType, BaseType
from ..lib import BaseHandler
from ...exceptions import OpenFileError
from ...pycompat import suppress

from collections import OrderedDict

from enstools.io import read


with suppress(ImportError):
    from xarray import Dataset
    def attrs(var):
        return var.attrs


class GRIBHandler(BaseHandler):

    """A simple handler for GRIB files.
    Here's a standard dataset for testing sequential data:
    """

    __version__ = get_distribution("pydap").version
    extensions = re.compile(r"^.*\.(grib|grib2)$", re.IGNORECASE)

    def __init__(self, filepath):
        BaseHandler.__init__(self)

        self.filepath = filepath
        try:
            print("read file ...")
            source = read(self.filepath, decode_times=False)
            source["time"] = [0]
            self.additional_headers.append(('Last-modified',
                                           (formatdate(
                                            time.mktime(
                                                time.localtime(
                                                    os.stat(filepath)
                                                    [ST_MTIME])
                                                    )))))

            # shortcuts
            vars = source.variables
            dims = source.dims

            # build dataset
            name = os.path.split(filepath)[1]
            self.dataset = DatasetType(name,
                                       attributes=dict(
                                                  NC_GLOBAL=attrs(source)))
            for dim in dims:
                if dims[dim] is None:
                    self.dataset.attributes['DODS_EXTRA'] = {
                        'Unlimited_Dimension': dim,
                    }
                    break

            # add grids
            grids = [var for var in vars if var not in dims]
            for grid in grids:
                print("attrs", attrs(vars[grid]))
                self.dataset[grid] = GridType(grid, attrs(vars[grid]))
                # add array
                self.dataset[grid][grid] = BaseType(grid,
                                                    LazyVariable(
                                                        source,
                                                        grid,
                                                        grid,
                                                        self.filepath),
                                                    source[grid].dims,
                                                    attrs(vars[grid]))
                # add maps
                for dim in vars[grid].dims:
                    print("dim", dim)
                    try:
                        data = vars[dim].values
                        attributes = attrs(vars[dim])
                    except KeyError:
                        data = np.arange(dims[dim], dtype='i')
                        attributes = None
                    self.dataset[grid][dim] = BaseType(dim, data,
                                                       None,
                                                       attributes)

                # add dims
                for dim in dims:
                    try:
                        data = vars[dim].values
                        attributes = attrs(vars[dim])
                    except KeyError:
                        print("here", dim)
                        data = np.arange(dims[dim], dtype='i')
                        attributes = None

                    self.dataset[dim] = BaseType(dim, data,
                                                 None,
                                                 attributes)
        except Exception as exc:
            raise
            message = 'Unable to open file %s: %s' % (filepath, exc)
            raise OpenFileError(message)


class LazyVariable:
    def __init__(self, source, name, path, filepath):
        self.filepath = filepath
        self.path = path
        var = source[self.path]
        self.dimensions = var.dims
        self.dtype = np.dtype(var.dtype)
        self.datatype = var.dtype
        self.ndim = len(var.dims)
        self._shape = var.shape
        self._reshape = var.shape
        self.scale = True
        self.name = name
        self.size = np.prod(self.shape)
        self._attributes = var.attrs
        return

    def chunking(self):
        return 'contiguous'

    def filters(self):
        return None

    def get_var_chunk_cache(self):
        raise NotImplementedError('get_var_chunk_cache is not implemented')
        return

    def ncattrs(self):
        return self._attributes

    def getncattr(self, attr):
        return self._attributes[attr]

    def __getattr__(self, name):
        # from netcdf4-python
        # if name in _private_atts, it is stored at the python
        # level and not in the netCDF file.
        if name.startswith('__') and name.endswith('__'):
            # if __dict__ requested, return a dict with netCDF attributes.
            if name == '__dict__':
                names = self.ncattrs()
                values = []
                for name in names:
                    values.append(self._attributes[name])
                return OrderedDict(zip(names, values))
            else:
                raise AttributeError
        else:
            return self.getncattr(name)

    def getValue(self):
        return self[...]

    def __array__(self):
        return self[...]

    def __getitem__(self, key):
        source = read(self.filepath)
        # Avoid applying scale_factor, see
        # https://github.com/pydap/pydap/issues/190
        return (np.asarray(source[self.path][key])
                .astype(self.dtype).reshape(self._reshape))

    def reshape(self, *args):
        if len(args) > 1:
            self._reshape = args
        else:
            self._reshape = args
        return self

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of unsized object')
        else:
            return self.shape[0]

    def _getdims(self):
        return self.dimensions


if __name__ == "__main__":
    import sys
    from werkzeug.serving import run_simple

    application = NetCDFHandler(sys.argv[1])
    run_simple('localhost', 8001, application, use_reloader=True)
