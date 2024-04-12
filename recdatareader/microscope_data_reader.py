# Tools to read data from the microscopes
import logging
from pathlib import Path
import dask.array
import numpy as np
from typing import Union # from pthon 3.10 it could be | instead of Union
from packaging import version

class MicroscopeDataReader:
    """
    Reads data from microscope data sets. The folder can be a NDTiff data store or a MMStack data store.
    If the folder is a NDTiff datastore, the data is read using the ndtiff package else tifffile is used.
    
    If a folder is passed, specific naming conventions are assumed.
    For ndtiff (if force_tifffile is True):
        first_tiff_file = self.directory_path.name + '_NDTiffStack.tif'
    and for MMStack:
        first_tiff_file = self.directory_path.name + '_MMStack.ome.tif'
    
    Args:
        path (Path or str): Path to the file or directory containing the data
        force_tifffile (bool, optional): If True, the file is read with tifffile. Defaults to False.
        is_btf (bool, optional): If True, the file is a BTF file. Defaults to False.
        btf_num_slices (int, optional): If is_btf is True, the number of slices of the BTF file have to be specified. Defaults to None.
    """
    def __init__(self, path: Union[Path,str], force_tifffile: bool = False, is_btf: bool = False, btf_num_slices: int = None):
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
        self._btf_num_slices = btf_num_slices
        self._is_btf: bool = is_btf
        self._is_ndtiff: bool = False
        self._is_tiffile: bool = False
        self._data_store = None
        if is_btf:
            self._read_btf_tifffile()
        else:
            self._open_dataset()
        
        
    
    def __del__(self):
        self.logger.info(f"Closing Microscope Data Reader")
        if self._data_store is not None:
            self._data_store.close()
    
    def _check_directory_path(self, directory_path: Union[Path,str]) -> None:
         # Check if file_path is of type pathlib.Path
        if isinstance(directory_path, Path):
            self.directory_path = directory_path
        elif isinstance(directory_path, str):
            self.directory_path = Path(directory_path)
        else:
            self.logger.error("file_path is not of type pathlib.Path or str")
            raise TypeError("file_path is not of type pathlib.Path or str")
         # Check if file or path exists:
        if not self.directory_path.exists():
            self.logger.error(f"File {self.directory_path} does not exist")
            raise FileNotFoundError(f"File {self.directory_path} does not exist")
        # check if self.filename is a file or a folder:
        if not self.directory_path.is_dir():
            self.first_tiff_file = self.directory_path.name
            self.directory_path = self.directory_path.parent
    
    def _open_dataset(self) -> None:
        """
        Opens the dataset. Also contains logic for finding first file if a directory is passed.
        """
        self.logger.info(f"Reading Dataset from: {self.directory_path}")
        if (self.directory_path/'NDTiff.index').exists():
            self.logger.info(f"Found NDTiff.index file in {self.directory_path}")
            self._is_btf = False
            self._is_tiffile = False
            self._is_ndtiff = True
            if self._force_tifffile:
                self.logger.info(f"Force reading with tiffile: {self.directory_path}")
                if self.first_tiff_file is None:
                    self.first_tiff_file = self.directory_path.name + '_NDTiffStack.tif'
                self._read_tifffile()
                return
            self._read_ndtiff()
            return
        if self.first_tiff_file is None:
            self.first_tiff_file = self.directory_path.name + '_MMStack.ome.tif'
        if (self.directory_path / self.first_tiff_file).exists():
            self.logger.info(f"Found {self.directory_path}/{self.first_tiff_file} file in {self.directory_path}")
            self._is_btf = False
            self._is_tiffile = True
            self._is_ndtiff = False
            self._read_tifffile()
            return
        else:
            self.logger.error(f"Could not find {self.directory_path}/{self.first_tiff_file} file in {self.directory_path}")
            raise FileNotFoundError(f"Could not find {self.directory_path}/{self.first_tiff_file} file in {self.directory_path}")
            
    def _read_ndtiff(self):
        self.logger.info(f"Reading data from {self.directory_path} as ndtiff file")
        from ndtiff import Dataset
        self._data_store = Dataset(str(self.directory_path))
        self._dask_array = self._data_store.as_array()
        self.logger.info(f"Data store: {self._data_store}")
        self.logger.info(f"dask array dimensions {self.axis_order}: {self._dask_array.shape}")
        
    def _read_tifffile(self):
        filepath = self.directory_path / self.first_tiff_file
        if not filepath.exists():
            self.logger.error(f"Could not find {self.first_tiff_file} file in {self.directory_path}")
            raise FileNotFoundError(f"Could not find {self.first_tiff_file} file in {self.directory_path}")
        self.logger.info(f"Reading data from {self.directory_path} as MMStack file")
        import tifffile as tff
        if version.parse(tff.__version__) < version.parse(self._tifffile_version):
            self.logger.error(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
            raise ImportError(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
        self._data_store = tff.TiffFile(filepath, mode='r')
        if not self._data_store.is_micromanager:
            self.logger.error(f"File {filepath} is not a Micromanager file")
        if self._is_tiffile and not self._data_store.is_mmstack:
            self.logger.error(f"File {filepath} is not a MMStack file")
        if self._is_ndtiff and not self._data_store.is_ndtiff:
            self.logger.error(f"File {filepath} is not a NDTiff file")
        axes = self._data_store.series[0].axes
        dask_array = dask.array.from_zarr(self._data_store.aszarr())
        self._tff_dask_array = dask_array
        # expected axis order: TRZCYX
        # reorder axis to (R)PTCZYX like ndtiff
        self._fix_axis_order_and_shape(axes, dask_array)
        self.logger.info(f"Data store: {self._data_store}")
        self.logger.info(f"dask array dimensions: {self._dask_array.shape}")
    
    def _read_btf_tifffile(self):
        filepath = self.directory_path / self.first_tiff_file
        if not filepath.exists():
            self.logger.error(f"Could not find {self.first_tiff_file} file in {self.directory_path}")
            raise FileNotFoundError(f"Could not find {self.first_tiff_file} file in {self.directory_path}")
        self.logger.info(f"Reading data from {self.directory_path} as btf file")
        import tifffile as tff
        if version.parse(tff.__version__) < version.parse(self._tifffile_version):
            self.logger.error(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
            raise ImportError(f"tifffile version {tff.__version__} is not supported. Please update to version {self._tifffile_version} or higher")
        self._data_store = tff.TiffFile(filepath, mode='r', is_ome=False)
        dask_array = dask.array.from_zarr(self._data_store.aszarr())
        if not len(dask_array.shape) == 3:
            self.logger.error(f"Expected 3D data [t,y,x], got {len(dask_array.shape)}D data")
            raise ValueError(f"Expected 3D data [t,y,x], got {len(dask_array.shape)}D data")
        if self._btf_num_slices is None:
            self.logger.warning(f"Number of slices in BTF file is not specified")
            self._read_MMStack_metadata_file_num_slices()
        if dask_array.shape[0] % self._btf_num_slices > 0:
            self.logger.error(f"Number of slices doesen't mach the timepoints in the BTF file")
            raise ValueError(f"Number of slices doesen't mach the timepoints in the BTF file")
        dask_array = dask_array.reshape((dask_array.shape[0]//self._btf_num_slices, self._btf_num_slices, dask_array.shape[1], dask_array.shape[2]))
        dask_array = dask.array.expand_dims(dask_array, axis=(0,2))
        self._dask_array = dask_array
        self._is_tiffile = True
        self._is_ndtiff = False
        self.logger.info(f"Data store: {self._data_store}")
        self.logger.info(f"dask array dimensions: {self._dask_array.shape}")
    
    def _read_MMStack_metadata_file_num_slices(self):
        import json
        my_path = Path(self.directory_path).glob("*_MMStack_metadata.txt")
        if all(False for _ in my_path):
            self.logger.error(f"Could not find *_MMStack_metadata.txt file in {self.directory_path}")
            raise FileNotFoundError(f"Could not find *_MMStack_metadata.txt file in {self.directory_path}")
        my_path = Path(self.directory_path).glob("*_MMStack_metadata.txt")
        for file in my_path:
            z_slices = 0
            self.logger.info(f"Reading metadata file {file}")
            with open(file, 'r') as f:
                json_file = json.load(f)
                for key in json_file.keys():
                    if "FrameKey" in key:
                        arr = key.split("-")
                        #print(arr)
                        new_z = int(arr[-1])
                        if new_z >= z_slices:
                            z_slices = new_z
                        else:
                            break
            self._btf_num_slices = z_slices + 1
            self.logger.info(f"Number of slices in {file.name}: {self._btf_num_slices}")
            return
        self.logger.error(f"Could not find number of slices in metadata file")
        raise FileNotFoundError(f"Could not find number of slices in metadata file")
    
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
                self.logger.debug(f"swap axis {org_axis_position[i]} with {i}")
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
        Reads a single y,x image from the data set. The image is selected by the position, time, channel and z values.
        
        Args:
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            z (int, optional): z-axis. Defaults to 0.
            
        Returns:
            np.array: xy image
        """
        if self._is_ndtiff:
            return self._data_store.read_image(position = position, time = time, channel=channel, z=z)
        if self._is_tiffile:
            return self._dask_array[position, time, channel, z, :, :].compute()
        return None
    
    def get_frame(self, position: int = 0, time: int = 0, channel: int = 0, z: int = 0) -> np.array:
        """
        Reads a single y,x image from the data set.
            The image is selected by the position, time, channel and z values.
        
        Args:
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            z (int, optional): z-axis. Defaults to 0.
            
        Returns:
            np.array: xy image
        """
        return self.read_image(position, time, channel, z)
    
    def read_single_volume(self, position: int = 0, time: int = 0, channel: int = 0) -> np.array:
        """
        Read a single volume from the data set
            The volume is selected by the position, time and channel values.
        
        Args:
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            
        Returns:
            np.array: volume[z,y,x]
        """
        return self._dask_array[position, time, channel, :, :, :].compute()
    
    def get_single_volume(self, position: int = 0, time: int = 0, channel: int = 0) -> np.array:
        """
        Read a single volume from the data set
            The volume is selected by the position, time and channel values.
        
        Args:
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            
        Returns:
            np.array: volume[z,y,x]
        """
        return self.read_single_volume(position, time, channel)
    
    def get_axis_order(self) -> list:
        """
        Returns the axis order of the dask array

        Returns:
            list: axis order
        """
        return self.axis_order
    
    def get_axis_string(self) -> str:
        """
        Returns the axis order of the daks array as string
        
        Returns:
            str: axis order
        """
        return self.axis_string
    
    def get_data_shape(self) -> tuple:
        """
        Returns the shape of the data set [position, time, channel, z, y, x]
        
        Returns:
            tuple: shape
        """
        return self._dask_array.shape
    
    def get_image_shape(self) -> tuple:
        """
        Returns the shape of the images [y, x]
        
        Returns:
            tuple: shape
        """
        return self._dask_array.shape[-2:]
    
    def get_volume_shape(self) -> tuple:
        """
        Returns the shape of the volumes [z, y, x]
        
        Returns:
            tuple: shape
        """
        return self._dask_array.shape[-3:]
    
    def get_number_of_positions(self) -> int:
        """
        Returns the amount of positions
        
        Returns:
            int: amount of positions
        """
        return self._dask_array.shape[0]
    
    def get_number_of_timepoints(self) -> int:
        """
        Returns the amount of timepoints
        
        Returns:
            int: amount of timepoints
        """
        return self._dask_array.shape[1]
    
    def get_number_of_channels(self) -> int:
        """
        Returns the amount of channels
        
        Returns:
            int: amount of channels
        """
        return self._dask_array.shape[2]
    
    def get_number_of_z_slices(self) -> int:
        """
        Returns the amount of slices
        
        Returns:
            int: amount of slices
        """
        return self._dask_array.shape[3]
    
    def get_total_number_of_frames(self) -> int:
        """
        Returns the total amount of frames
        
        Returns:
            int: total amount of frames
        """
        return self.dask_array.shape[0] * self.dask_array.shape[1] * self.dask_array.shape[2] * self.dask_array.shape[3]
    
    def get_number_of_volumes(self) -> int:
        """
        Returns the amount of volumes
        
        Returns:
            int: amount of volumes
        """
        return self._dask_array.shape[0] * self._dask_array.shape[1] * self._dask_array.shape[2]
    
    def get_data_type(self) -> str:
        """
        Returns the data type of the data set
        
        Returns:
            str: data type
        """
        return self._dask_array.dtype
    
    def open_in_napari(self):
        """
        Opens the data set in napari using view_image. This assumes the image is floats (not ints, like segmentation)
        """
        import napari
        viewer = napari.view_image(self._dask_array, multiscale=False, rgb=False, axis_labels=self.axis_order)
        return viewer

    def _generate_ome_metadata(self):
        from omexmlClass import OMEXML
        my_ome_mxl = OMEXML()
        my_ome_mxl.set_image_count(1000)
        my_ome_mxl.Plane
        
    @property
    def dask_array(self):
        return self._dask_array