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
        self.finish()
        if self.verbose >= 1:
            self.logger.info(f"Closing Microscope Data Writer")
        self.directory_path: Path = None
        if self._data_store is not None:
            self._data_store.close()
        self._data_store = None
    
    def finish(self):
        if self.verbose >= 1:
            self.logger.info(f"Finishing Microscope Data Writer")
        self._data_store.finish()
    
    def get_image_coordinates_list(self):
        return self._data_store.get_image_coordinates_list()
    
    def read_image(self, position: int = 0, time: int = 0, channel: int = 0, z: int = 0):
        return self._data_store.read_image(channel=channel, z=z, time=time, position=position)
    
    def read_image_metadata(self, position: int = 0, time: int = 0, channel: int = 0, z: int = 0):
        return self._data_store.read_metadata(channel=channel, z=z, time=time, position=position)
    
    def get_summary_metadata(self):
        return self.summary_metadata
    
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
        
    def write_image(self, image: np.array, position: int = 0, time: int = 0, channel: int = 0, z: int = 0, image_metadata: dict = None) -> None:
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
        
        basic_metadata = {'ElapsedTimeWriter_ms': (datetime.time() - self.init_date_time).total_seconds() * 1000}
        if image_metadata is None:
            if self.verbose >= 1:
                self.logger.warning("No metadata provided!")
            basic_metadata['MicroscopeDataWriter'] = 'no metadata provided!'
            image_metadata = basic_metadata
        else:
            image_metadata = basic_metadata | image_metadata
        
        self._data_store.put_image(image_coordinates, image, image_metadata)
         
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
    
    
