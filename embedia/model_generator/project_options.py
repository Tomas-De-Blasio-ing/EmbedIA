from enum import Enum, IntEnum

MODEL_DATA_TYPE_SIZES = (32, 32, 16, 8, 8, 8, 32, 32, 16)  # Bit sizes for each data type

class ModelDataType(Enum):
    """
    Enumeration of supported data types for model quantization and inference.
    Each type has an associated bit size and string name.
    """
    FLOAT             = 0
    FIXED32           = 1
    FIXED16           = 2
    FIXED8            = 3
    QUANT8            = 4
    FULL_QUANT8       = 5
    BINARY            = 6
    BINARY_FIXED32    = 7
    BINARY_FLOAT16    = 8


    @property
    def size(self):
        """Returns the bit size of the data type."""
        return MODEL_DATA_TYPE_SIZES[self.value]

    @property
    def lname(self):
        """Returns the lowercase name of the data type."""
        return self.name.lower()


class ModelMicro(Enum):
    """
    Target microcontroller platforms for hardware-specific optimizations.
    """
    GENERIC = 0
    ESP32   = 1

    @property
    def lname(self):
        """Returns the string name of the microcontroller."""
        return self.name.lower()


class ProjectType(Enum):
    """
    Type of project to generate (affects file structure and code style).
    """
    C         = 0
    CPP       = 1
    ARDUINO   = 2
    CODEBLOCK = 3  # For Code::Blocks IDE projects


class ProjectFiles(Enum):
    """
    Files to include in the exported project.
    Uses IntEnum to support bitmask operations if needed.
    """
    LIBRARY = 1  # embedia library files
    MAIN    = 2  # main application file (e.g., main.c)
    MODEL   = 4  # model file (e.g., model_data.c)

    @classmethod
    def ALL(cls):
        """Returns a set containing all project file types."""
        return {cls.LIBRARY, cls.MAIN, cls.MODEL}


class DebugMode(IntEnum):
    """
    Level of debug information to include in the generated code.
    Negative values are allowed (e.g., -1 for special behavior).
    """
    DISCARD   = -1  # Discard all debug info
    DISABLED  = 0   # No debug output
    HEADERS   = 1   # Include debug headers
    DATA      = 2   # Include full data dumps


class BinaryBlockSize(Enum):
    """
    Block size (in bits) for packing binary weights in binary neural networks.
    """
    Bits8  = 0
    Bits16 = 1
    Bits32 = 2
    Bits64 = 3


class UnimplementedLayerAction(Enum):
    """
    Action to take when an unimplemented layer is found during model export.
    """
    FAILURE        = 0  # Raise an error and stop
    IGNORE_ALL     = 1  # Skip the layer silently
    IGNORE_KNOWN   = 2  # Skip only known unimplemented layers (with warning)


class ProjectOptions:
    """
    Configuration options for project generation.
    This class holds all user-defined settings for the export process.
    """
    def __init__(self):
        self.embedia_folder = None           # embedia source folder
        self.embedia_output_subfloder = ''   # subfolder in output folder to copy embedia files
        self.project_type = ProjectType.C    # project type to export
        self.micro = ModelMicro.GENERIC      # microcontroller for hardware optimization
        self.data_type = ModelDataType.FLOAT # data type for data storage
        self.baud_rate = 9600                # Arduino Only. Set Serial device speed
        self.example_data = None             # list of examples to include in project
        self.example_labels = None           # list of labels for examples (classification)
        self.files = ProjectFiles.ALL()      # set of files to export library, main or model
        self.debug_mode = DebugMode.DISABLED # debug info to include and what to show
        self.clean_output = False            # clear output folder before export (use carefully)
        self.preprocessing = None            # preprocessing objects to add before start inference
        self.tamano_bloque = BinaryBlockSize.Bits8 # block size for binary nets
        self.on_unimplemented_layer = UnimplementedLayerAction.IGNORE_KNOWN # error action when find an unimplemented layer
        self.output_subfolder = 'embedia'    # name of folder to store all embedia files