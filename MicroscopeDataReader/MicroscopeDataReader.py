# Tools to read data from the microscopes
import logging
from pathlib import Path
import dask.array
import numpy as np
from typing import Union # from pthon 3.10 it could be | instead of Union
from packaging import version

class MicroscopeDataReader:
    def __init__(self, path: Union[Path,str], force_tifffile: bool = False):
        """
        Reads data from a microscope data file. The file can be a NDTiff file or a MMStack file.
        If the file is a NDTiff file, the data is read using the ndtiff package.
        
        Args:
            path (Path or str): Path to the file or directory containing the data
            force_tifffile (bool, optional): If True, the file is read with tifffile. Defaults to False.
        """
        self._force_tifffile = force_tifffile
        self.logger = logging.getLogger(__name__)
        # make sure logging works:
        if not self.logger.hasHandlers():
            log_format = '%(asctime)s.%(msecs)03d | %(levelname)-8s |\033[1;36m %(message)s\033[0m \033[5m | %(name)s \033[0m\033[3m- %(funcName)s -\033[0m \033[1mLine %(lineno)d\033[0m'
            logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
        self._tifffile_version = '2023.7.10'
        self.directory_path: Path = None
        self.first_tiff_file: str = None
        self._tff_dask_array: dask.array = None
        self._dask_array: dask.array = None
        self.axis_order = ['position', 'time', 'channel', 'z', 'y', 'x']
        self.axis_string = 'PTCZYX'
        self._axis_string_tifffile = 'RTCZYX'
        self._check_directory_path(path)
        self.is_ndtiff: bool = False
        self.is_mm_stack: bool = False
        self._data_store = None
        self._open_dataset()
        
        
    
    def __del__(self):
        self.logger.info(f"Closing Microscope Data Reader")
        self._data_store.close()
    
    def _check_directory_path(self, directory_path: Union[Path,str]) -> None:
         # Check if file_path is of type pathlib.Path
        if isinstance(directory_path, Path):
            self.directroy_path = directory_path
        elif isinstance(directory_path, str):
            self.directroy_path = Path(directory_path)
        else:
            self.logger.error("file_path is not of type pathlib.Path or str")
            raise TypeError("file_path is not of type pathlib.Path or str")
         # Check if file or path exists:
        if not self.directroy_path.exists():
            self.logger.error(f"File {self.directroy_path} does not exist")
            raise FileNotFoundError(f"File {self.directroy_path} does not exist")
        # check if self.filename is a file or a folder:
        if not self.directroy_path.is_dir():
            self.first_tiff_file = self.directroy_path.name
            self.directroy_path = self.directroy_path.parent
    
    def _open_dataset(self) -> None: 
        self.logger.info(f"Reading Dataset from: {self.directroy_path}")
        if (self.directroy_path/'NDTiff.index').exists():
            self.logger.info(f"Found NDTiff.index file in {self.directroy_path}")
            self.is_mm_stack = False
            self.is_ndtiff = True
            if self._force_tifffile:
                self.logger.info(f"Force reading with tiffile: {self.directroy_path}")
                if self.first_tiff_file is None:
                    self.first_tiff_file = self.directroy_path.name + '_NDTiffStack.tif'
                self._read_tifffile()
                return
            self._read_ndtiff()
            return
        if self.first_tiff_file is None:
            self.first_tiff_file = self.directroy_path.name + '_MMStack.ome.tif'
        if (self.directroy_path / self.first_tiff_file).exists():
            self.logger.info(f"Found {self.directroy_path}/{self.first_tiff_file} file in {self.directroy_path}")
            self.is_mm_stack = True
            self.is_ndtiff = False
            self._read_tifffile()
            return
        else:
            self.logger.error(f"Could not find {self.directory_path}/{self.first_tiff_file} file in {self.directroy_path}")
            raise FileNotFoundError(f"Could not find {self.directory_path}/{self.first_tiff_file} file in {self.directroy_path}")
            
    def _read_ndtiff(self):
        self.logger.info(f"Reading data from {self.directroy_path} as ndtiff file")
        from ndtiff import Dataset
        self._data_store = Dataset(self.directroy_path)
        self._dask_array = self._data_store.as_array()
        self.logger.info(f"Data store: {self._data_store}")
        self.logger.info(f"dask array dimensions [position?,t,channel?,z,y,x]: {self._dask_array.shape}")
        
    def _read_tifffile(self):
        filepath = self.directroy_path / self.first_tiff_file
        if not filepath.exists():
            self.logger.error(f"Could not find {self.first_tiff_file} file in {self.directroy_path}")
            raise FileNotFoundError(f"Could not find {self.first_tiff_file} file in {self.directroy_path}")
        self.logger.info(f"Reading data from {self.directroy_path} as MMStack file")
        import tifffile as tff
        if version.parse(tff.__version__) < version.parse(self._tifffile_version):
            self.logger.error(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
            raise ImportError(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
        self._data_store = tff.TiffFile(filepath, mode='r')
        if not self._data_store.is_micromanager:
            self.logger.error(f"File {filepath} is not a Micromanager file")
        if self.is_mm_stack and not self._data_store.is_mmstack:
            self.logger.error(f"File {filepath} is not a MMStack file")
        if self.is_ndtiff and not self._data_store.is_ndtiff:
            self.logger.error(f"File {filepath} is not a NDTiff file")
        axes = self._data_store.series[0].axes
        dask_array = dask.array.from_zarr(self._data_store.aszarr())
        self._tff_dask_array = dask_array
        # expected axis order: TRZCYX
        # reorder axis to (R)PTCZYX like ndtiff
        self._fix_axis_order_and_shape(axes, dask_array)
        self.logger.info(f"Data store: {self._data_store}")
        self.logger.info(f"dask array dimensions: {self._dask_array.shape}")
    
    def _fix_axis_order_and_shape(self, axes: str, dask_array: dask.array):
        # fix axis order to be PTCZYX
        self.logger.info(f"Fixing axis order")
        org_axis_position = [None,None,None,None,None,None]
        # checking for axis order RTCZYX:
        for i, org in enumerate(self._axis_string_tifffile):
            org_axis_position[i] = axes.find(org)
            self.logger.debug(f"Axis {i}, {org}, is at position {org_axis_position[i]}")
        self.logger.info(f"Org axis order, position of PTCZYX: {org_axis_position}")
        # reorder axis to PTCZYX
        for i in range(len(org_axis_position)):
            self.logger.debug(f"Axis {i}, {self.axis_order[i]}, is at position {org_axis_position[i]}")
            if org_axis_position[i] == -1:
                self.logger.debug(f"Adding axis {i}, {self.axis_order[i]}")
                dask_array = dask.array.expand_dims(dask_array, i)
                org_axis_position[i] = i
                # add +1 to all elements of org_axis_position if they are >= 0 (not -1) to take into account the added axis
                org_axis_position[i+1:] = [x+1 if x >= 0 else x for x in org_axis_position[i+1:]]
            elif not org_axis_position[i] == i:
                self.logger.debug(f"swapp axis {org_axis_position[i]} with {i}")
                dask_array = dask.array.moveaxis(dask_array, org_axis_position[i], i)
                # also flip axis in org_axis_position
                old_axis = org_axis_position[i]
                org_axis_position[i] = i
                org_axis_position[old_axis] = old_axis
            else:
                self.logger.debug(f"Axis {i}, {self.axis_order[i]}, nothing to do")
        
        self.logger.info(f"New axis order [p,t,c,z,y,x]: {org_axis_position}")
        self._dask_array = dask_array
        
    def read_image(self, position: int = 0, time: int = 0, channel: int = 0, z: int = 0) -> np.array:
        """
        Reads an single y,x image from the data set. The image is selected by the position, time, channel and z values.
        
        Args:
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            z (int, optional): z-axis. Defaults to 0.
            
        Returns:
            np.array: xy image
        """
        if self.is_ndtiff:
            return self._data_store.read_image(position = position, time = time, channel=channel, z=z)
        if self.is_mm_stack:
            return self._dask_array[position, time, channel, z, :, :].compute()
        return None
    
    def get_axes_order(self) -> list:
        """
        Returns the axis order of the dask array

        Returns:
            list: axis order
        """
        return self.axis_order
    
    def get_axes_string(self) -> str:
        """
        Returns the axis order of the daks array as string
        
        Returns:
            str: axis order
        """
        return self.axis_string
    
    def open_in_napari(self):
        """
        Opens the data set in napari
        """
        import napari
        viewer = napari.view_image(self._dask_array, multiscale=False, rgb=False, axis_labels=self.axis_order)
        return viewer
    
    @property
    def dask_array(self):
        return self._dask_array