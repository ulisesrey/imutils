from loguru import logger
from pathlib import Path
from datetime import datetime
import time
import dask.array
import numpy as np
from typing import Union # from pthon 3.10 it could be | instead of Union
from packaging import version
from ndstorage import NDTiffDataset

class MicroscopeDataWriter:
    """
    Writes data for microscope data sets as ndtiff.
    Metadata is stored in a header and as seperate files.
    Trys to keep the micromanager metadata schema.
    """
    def __init__(self, path: Union[Path,str], dataset_name: str, add_date_time: bool = True, summary_metadata: dict = None, verbose: int = 1):
        """
        Writes data for microscope data sets as ndtiff.
        Metadata is stored in a header and as seperate files.
        Trys to keep the micromanager metadata schema.
        
        Args:
            path (Path or str): Path to the file or directory containing the data
            verbose (int, optional): Verbosity level of logger messages. Defaults to 1.
        """
        self.verbose = verbose
        self.init_date_time = datetime.today()
        if add_date_time:
            self.dataset_name = f"{self.init_date_time.strftime('%Y-%m-%d_%H-%M')}_{dataset_name}"
        else:
            self.dataset_name = dataset_name
        self.logger = logger.bind(classname=self.__class__.__name__)
        self.directory_path: Path = None
        self.axis_order = ['position', 'time', 'channel', 'z', 'y', 'x']
        self.axis_string = 'PTCZYX'
        self._check_directory_path(path)
        self._data_store = None
        basic_metadata = {'MicroscopeDataWriter': 'metadata provided', 'Package': 'imutils',
                          'Library': 'ndstorage', 'Date': self.init_date_time.strftime("%d.%m.%Y"),
                          'TimeCreated': self.init_date_time.strftime("%H:%M:%S")}
        if summary_metadata is None:
            if self.verbose >= 1:
                self.logger.warning("No metadata provided!")
            basic_metadata['MicroscopeDataWriter'] = 'no metadata provided!'
            self.summary_metadata = basic_metadata
        else:
            self.summary_metadata = basic_metadata | summary_metadata        
        self._init_dataset()
        
    def __enter__(self):
        if self.verbose >= 1:
            self.logger.warning("Use the build in methods to write data and metadata to file")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.verbose >= 1:
            self.logger.warning(f"Your file is written and cant be changed anymore!")
        self.close()
    
    def close(self):
        if self.verbose >= 1:
            self.logger.info(f"Closing Microscope Data Writer")
        self.directory_path: Path = None
        self.axis_order = ['position', 'time', 'channel', 'z', 'y', 'x']
        self.axis_string = 'PTCZYX'
        self._axis_string_tifffile = 'RTCZYX'
        if self._data_store is not None:
            self._data_store.close()
        self._data_store = None
    
    def _check_directory_path(self, directory_path: Union[Path,str], overwrite: bool) -> None:
         # Check if file_path is of type pathlib.Path
        if isinstance(directory_path, Path):
            self.directory_path = directory_path
        elif isinstance(directory_path, str):
            self.directory_path = Path(directory_path)
        else:
            if self.verbose >= 1:
                self.logger.error("file_path is not of type pathlib.Path or str")
            raise TypeError("file_path is not of type pathlib.Path or str")
        # Check if file or path exists:
        # the ndstore does this for us
        # final_path = self.directory_path / self.dataset_name
        # if final_path.exists() and not overwrite:
        #     if self.verbose >= 1:
        #         self.logger.error(f"Dataset {self.directory_path} does exist!")
        #     raise FileExistsError(f"Dataset {self.directory_path} does exist!")
    
    def _init_dataset(self) -> None:
        """
        Initializes the dataset.
        """
        if self.verbose >= 1:
            self.logger.info(f"Init Dataset at: {self.directory_path}")
        self._data_store = NDTiffDataset(dataset_path=self.directory_path, name=self.dataset_name,
                                         summary_metadata=self.summary_metadata, writable=True)
        
    def write_image(self, image: np.array, position: int = 0, time: int = 0, channel: int = 0, z: int = 0) -> None:
        """
        Writes a single y,x image to the data set. The image is selected by the position, time, channel and z values.
        
        Args:
            image (np.array): xy image
            position (int, optional): position. Defaults to 0.
            time (int, optional): time. Defaults to 0.
            channel (int, optional): channel. Defaults to 0.
            z (int, optional): z-axis. Defaults to 0.
        """
        image_coordinates = {'position': position, 'time': time, 'channel': channel, 'z': z}
        basic_metadata = {'WriteTime_µs': time.time()}
        image_metadata = {'ImageCoordinates': image_coordinates}
        self._data_store.put_image(image_coordinates, image, image_metadata)
        
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
