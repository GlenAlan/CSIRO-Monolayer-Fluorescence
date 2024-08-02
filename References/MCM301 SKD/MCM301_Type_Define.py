from ctypes import *


class BoardStatusInfoStruct(Structure):
    """ Board information
        border_temperature: Border temperature
        cpu_temperature: Cpu temperature
        high_voltage: High voltage
        error_code: Slot error signal. 0x04:slot 4 error; 0x05:slot 5 error; 0x06:slot 6 error
    """
    _fields_ = [("border_temperature", c_double),
                ("cpu_temperature", c_double),
                ("high_voltage", c_double),
                ("error_code", c_byte)
                ]


class StageParamsInfoStruct(Structure):
    """ The parameters for this stage
        counts_per_unit: The number of encoder counts per stepper motor step
        nm_per_count: The number of nanometers per encoder count.
        minimum_position: The smallest encoder value of the stage when homed
        maximum_position: The largest encoder value of the stage when homed
        maximum_speed: The Upper limit for the speed of the stepper
        maximum_acc: The Upper limit for the acceleration of the stepper
    """
    _fields_ = [("counts_per_unit", c_uint),
                ("nm_per_count", c_float),
                ("minimum_position", c_uint),
                ("maximum_position", c_uint),
                ("maximum_speed", c_double),
                ("maximum_acc", c_double)
                ]


class EFSHWInfoStruct(Structure):
    """ The state information of EFS
        available: available signal.0:available 1:unavailable
        version: EFS version
        page_size: The size of a page in bytes.
        pages_supported: The number of pages in the file system.
        maximum_files: The maximum number of files supported by the system
        files_remain: The number of files that can be allocated
        pages_remain: The number of pages remaining
    """
    _fields_ = [("available", c_byte),
                ("version", c_byte),
                ("page_size", c_uint16),
                ("pages_supported", c_uint16),
                ("maximum_files", c_uint16),
                ("files_remain", c_uint16),
                ("pages_remain", c_uint16)
                ]


class EFSFileInfoStruct(Structure):
    """ The information of file in EFS
        file_name: File name
        exist: File exist signal. 0 when the file does not exist
        owned: Indicates that the file is owned by the firmware
        attributes: File attribute.
            0x01:APT Read Allowed;
            0x02:APT Write Allowed;
            0x04:APT Delete Allowed;
            0x08:Firmware Read Allowed;
            0x10:Firmware Write Allowed;
            0x20:Firmware Delete Allowed;
        file_size: Length of the file in pages.
    """
    _fields_ = [("file_name", c_byte),
                ("exist", c_byte),
                ("owned", c_byte),
                ("attributes", c_byte),
                ("file_size", c_uint16)
                ]
