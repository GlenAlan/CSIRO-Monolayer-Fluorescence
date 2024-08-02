from MCM301_Type_Define import *


class MCM301:
    mcm301Lib = None
    isLoad = False

    @staticmethod
    def list_devices():
        """ List all connected mcm301 devices
        Returns:
           The mcm301 device list, each deice item is serialNumber/COM
        """
        str1 = create_string_buffer(10240)
        result = MCM301.mcm301Lib.List(str1, 10240)
        devicesStr = str1.value.decode("utf-8", "ignore").rstrip('\x00').split(',')
        length = len(devicesStr)
        i = 0
        devices = []
        devInfo = ["", ""]
        while i < length:
            str2 = devicesStr[i]
            if i % 2 == 0:
                if str2 != '':
                    devInfo[0] = str2
                else:
                    i += 1
            else:
                devInfo[1] = str2
                devices.append(devInfo.copy())
            i += 1
        return devices

    @staticmethod
    def load_library(path):
        MCM301.mcm301Lib = cdll.LoadLibrary(path)
        MCM301.isLoad = True

    def __init__(self):
        #lib_path = "./MCM301Lib_Win32.dll" OBSOLETE. Change to 64bit MB
        lib_path = "dlls/MCM301Lib_x64.dll"
        if not MCM301.isLoad:
            MCM301.load_library(lib_path)
        self.hdl = -1

    def open(self, serialNo, nBaud, timeout):
        """ Open MCM301 device
        Args:
            serialNo: serial number of MCM301 device
            nBaud: the bit per second of port
            timeout: set timeout value in (s)
        Returns: 
            non-negative number: hdl number returned Successful; negative number: failed.
        """
        ret = -1
        if MCM301.isLoad:
            ret = MCM301.mcm301Lib.Open(serialNo.encode('utf-8'), nBaud, timeout)
            if ret >= 0:
                self.hdl = ret
            else:
                self.hdl = -1
        return ret

    def is_open(self, serialNo):
        """ Check opened status of MCM301 device
        Args:
            serialNo: serial number of MCM301 device
        Returns: 
            0: MCM301 device is not opened; 1: MCM301 device is opened.
        """
        ret = -1
        if MCM301.isLoad:
            ret = MCM301.mcm301Lib.IsOpen(serialNo.encode('utf-8'))
        return ret

    def get_handle(self, serialNo):
        """ get handle of port
        Args:
            serialNo: serial number of the device to be checked.
        Returns: 
            -1:no handle  non-negative number: handle.
        """
        ret = -1
        if MCM301.isLoad:
            ret = MCM301.mcm301Lib.GetHandle(serialNo.encode('utf-8'))
        return ret

    def close(self):
        """ Close opened MCM301 device
        Returns: 
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            ret = MCM301.mcm301Lib.Close(self.hdl)
        return ret

    def get_error_state(self):
        """ Get the device error state
        Returns:
              0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            ret = MCM301.mcm301Lib.GetErrorState(self.hdl)
        return ret

    def set_chan_enable_state(self, slot, enable_state):
        """ Set enable or disable a stepper
        Args:
            slot: target slot (4,5,6)
            enable_state: 0 Disable the stepper.1 Enable the stepper.
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            state = c_byte(enable_state)
            ret = MCM301.mcm301Lib.SetChanEnableState(self.hdl, slot_val, state)
        return ret

    def get_chan_enable_state(self, slot, enable_state):
        """ Get the enabled status of stepper.
        Args:
            slot: target slot (4,5,6)
            enable_state:  0 Stepper is disabled. 1 Stepper is enabled.
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            state = c_byte(0)
            ret = MCM301.mcm301Lib.GetChanEnableState(self.hdl, slot_val, byref(state))
            enable_state[0] = state.value
        return ret

    def set_jog_params(self, slot, step_size):
        """ Set the jogging parameters for the slot card.
        Args:
            slot: target slot (4,5,6)
            step_size: jog step size
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            step_size_val = c_uint(step_size)
            ret = MCM301.mcm301Lib.SetJogParams(self.hdl, slot_val, step_size_val)
        return ret

    def get_jog_params(self, slot, jog_step_size):
        """ Get the jogging parameters for the slot card.
        Args:
            slot: target slot (4,5,6)
            jog_step_size: jog step size (encoder)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            step_size_val = c_uint(0)
            ret = MCM301.mcm301Lib.GetJogParams(self.hdl, slot_val, byref(step_size_val))
            jog_step_size[0] = step_size_val.value
        return ret

    def set_MOT_encounter(self, slot, encoder_count):
        """ Set the encoder count of the stepper to the provided value
        Args:
            slot: target slot (4,5,6)
            encoder_count: the encoder count of the stepper
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            encoder_count_val = c_uint(encoder_count)
            ret = MCM301.mcm301Lib.SetMOTEncCounter(self.hdl, slot_val, encoder_count_val)
        return ret

    def set_slot_title(self, slot, title, title_length):
        """ Set the title of slot.
        Args:
            slot: target slot (4,5,6)
            title: User-defined title for the slot
            title_length: the length of title (it should be larger than 0 and less than 16)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            title_val = create_string_buffer(title, title_length)
            ret = MCM301.mcm301Lib.SetSlotTitle(self.hdl, slot_val, byref(title_val), title_length)
        return ret

    def get_slot_title(self, slot, title, title_length):
        """ Get the title of slot.
        Args:
            slot: target slot (4,5,6)
            title: User-defined title for the slot
            title_length: the length of title(it should be larger than 16)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            title_val = create_string_buffer(title_length)
            ret = MCM301.mcm301Lib.GetSlotTitle(self.hdl, slot_val, byref(title_val), title_length)
            title[0] = title_val.value.decode("utf-8", "ignore").rstrip('\x00').replace("\r\n", "")
        return ret

    def set_system_dim(self, dim):
        """ Set the maximum brightness of the LEDs connected to the controller.
        Args:
            dim: 0<=D<=100
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            dim_val = c_byte(dim)
            ret = MCM301.mcm301Lib.SetSystemDim(self.hdl, dim_val)
        return ret

    def get_system_dim(self, dim):
        """ Get the maximum brightness of the LEDs connected to the controller.
        Args:
            dim: 0<=D<=100
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            dim_val = c_byte(0)
            ret = MCM301.mcm301Lib.GetSystemDim(self.hdl, byref(dim_val))
            dim[0] = dim_val.value
        return ret

    def set_soft_limit(self, slot, mode):
        """ Changes the rotational soft limits based on the current encoder position.
        Args:
            slot: target slot (4,5,6)
            mode: 1:set the counter-clockwise soft limit to the current encoder position; 2:set the clockwise high soft
            limit to the current encoder position; 3.removes both the high and low soft limit
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            mode_val = c_byte(mode)
            ret = MCM301.mcm301Lib.SetSoftLimit(self.hdl, slot_val, mode_val)
        return ret

    def set_EEPROM_PARAMS_soft_limit(self, slot):
        """ Set the software limit to EEPROM
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.SetEEPROMPARAMSSoftLimit(self.hdl, slot_val)
        return ret

    def chan_identify(self, slot):
        """ Sent to initiate the LED identification sequence on the controller(power LED) as well as any HID LEDs for
        controls mapped to the given slot.
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.ChanIdentify(self.hdl, slot_val)
        return ret

    def home(self, slot):
        """ Begin a homing movement
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.Home(self.hdl, slot_val)
        return ret

    def move_stop(self, slot):
        """ Stop any motion on this stepper.
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.MoveStop(self.hdl, slot_val)
        return ret

    def move_absolute(self, slot, target_encoder):
        """ Move the stepper to a specified encoder position.
        Args:
            slot: target slot (4,5,6)
            target_encoder:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            target_encoder_val = c_int(target_encoder)
            ret = MCM301.mcm301Lib.MoveAbsolute(self.hdl, slot_val, target_encoder_val)
        return ret

    def move_jog(self, slot, direction):
        """ Start a jog movement in the specified direction.
        Args:
            slot: target slot (4,5,6)
            direction: 0 Counter-Clockwise;1 Clockwise
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            direction_val = c_int(direction)
            ret = MCM301.mcm301Lib.MoveJog(self.hdl, slot_val, direction_val)
        return ret

    def erase_configuration(self, slot):
        """ Remove any custom configurations on a given slot card, falling back on the controller’s default
        configuration for the connected device(if one exists).
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.EraseConfiguration(self.hdl, slot_val)
        return ret

    def restart_board(self):
        """ Manually restart the board.
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            ret = MCM301.mcm301Lib.RestartBoard(self.hdl)
        return ret

    def convert_encoder_to_nm(self, slot, encoder_count, nm):
        """ Convert raw encoder to nm
        Args:
            slot: target slot (4,5,6)
            encoder_count:
            nm:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            encoder_count_val = c_int(encoder_count)
            nm_val = c_double(0)
            ret = MCM301.mcm301Lib.ConvertEncoderTonm(self.hdl, slot_val, encoder_count_val, byref(nm_val))
            nm[0] = nm_val.value
        return ret

    def convert_nm_to_encoder(self, slot, nm, encoder_count):
        """ Convert nm to raw encoder
        Args:
            slot: target slot (4,5,6)
            nm:
            encoder_count:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            nm_val = c_double(nm)
            encoder_count_val = c_int(0)
            ret = MCM301.mcm301Lib.ConvertnmToEncoder(self.hdl, slot_val, nm_val, byref(encoder_count_val))
            encoder_count[0] = encoder_count_val.value
        return ret

    def get_hardware_info(self, firmware_version, firmware_version_buffer_len, cpid_version, cpid_version_buffer_len):
        """ Get hardware information from the controller.
        Args:
            firmware_version: Stored in minor, interim, major order
            firmware_version_buffer_len: the firmware version buffer, it should be larger than 3
            cpid_version: Stored in major, minor order.
            cpid_version_buffer_len: >the CPID version buffer, it should be larger than 2
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            firmware_version_val = (c_byte * firmware_version_buffer_len)()
            cpid_version_val = (c_byte * cpid_version_buffer_len)()
            ret = MCM301.mcm301Lib.GetHardwareInfo(self.hdl, byref(firmware_version_val), firmware_version_buffer_len,
                                                   byref(cpid_version_val), cpid_version_buffer_len)
            for i in range(firmware_version_buffer_len):
                firmware_version[i] = firmware_version_val[i]
            for j in range(cpid_version_buffer_len):
                cpid_version[j] = cpid_version_val[j]
        return ret

    def get_mot_status(self, slot, current_encoder, status_bit):
        """  Get general information (i.e. status) of the stepper driver.
        Args:
            slot: target slot (4,5,6)
            current_encoder: For stages with an encoder, this represents the value from the encoder.For stages without
                             an encoder, this equals to the position divided by this stepper’s counts per unit.
            status_bit: 0x01:On Clockwise Hardware Limit Switch.0x02:On Counter-Clockwise Hardware Limit Switch.
                        0x04:On Clockwise Software Limit Switch.0x08:On Counter-Clockwise Software Limit Switch.
                        0x10:Moving Clockwise Stage is in motion.0x20:Moving Counter-Clockwise Stage is in motion.
                        0x40:Jogging Clockwise Stage is in motion.0x80:Jogging Counter-Clockwise Stage is in motion.
                        0x100:Motor Connected The motor has been recognized by the controller.
                        0x200:Homing Stage is in motion.Homed.
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            current_encoder_val = c_int(0)
            status_bit_val = c_uint(0)
            ret = MCM301.mcm301Lib.GetMotStatus(self.hdl, slot_val, byref(current_encoder_val), byref(status_bit_val))
            current_encoder[0] = current_encoder_val.value
            status_bit[0] = status_bit_val.value
        return ret

    def get_PNP_status(self, slot, status):
        """ Get the plug-and-play status of a given slot card
        Args:
            slot: target slot (4,5,6)
            status: 0x00:Normal.
                    0x01:No Device Connected.
                    0x02:General One-Wire Error.
                    0x04:Unknown One - Wire Version.
                    0x08:One-Wire Corruption.
                    0x10:Serial Number Mismatch.
                    0x20:Device Signature Not Allowed.
                    0x40:General Configuration Error.
                    0x80:Device Configuration Set Miss.
                    0x100:Configuration Struct Miss
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            status_val = c_uint(0)
            ret = MCM301.mcm301Lib.GetPNPStatus(self.hdl, slot_val, byref(status_val))
            status[0] = status_val.value
        return ret

    def get_board_status(self, border_status_struct):
        """ Get the temperature sensors, high-voltage input, and slot card error bits.
        Args:
            border_status_struct: the information of board
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            border_status_struct_val = BoardStatusInfoStruct()
            ret = MCM301.mcm301Lib.GetBoardStatus(self.hdl, byref(border_status_struct_val))
            border_status_struct[0] = [border_status_struct_val.border_temperature,
                                       border_status_struct_val.cpu_temperature,
                                       border_status_struct_val.high_voltage,
                                       border_status_struct_val.error_code]
        return ret

    def get_stage_params(self, slot, stage_params_info):
        """ Get stage parameters of this slot card
        Args:
            slot: target slot (4,5,6)
            stage_params_info: the stage parameters
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            stage_params_info_val = StageParamsInfoStruct()
            ret = MCM301.mcm301Lib.GetStageParams(self.hdl, slot_val, byref(stage_params_info_val))
            stage_params_info[0] = [stage_params_info_val.counts_per_unit, stage_params_info_val.nm_per_count,
                                    stage_params_info_val.minimum_position, stage_params_info_val.maximum_position,
                                    stage_params_info_val.maximum_speed, stage_params_info_val.maximum_acc]
        return ret

    def get_slot_device_type(self, slot, device_type, device_type_length):
        """ Get identifying information on a device.
        Args:
            slot: target slot (4,5,6)
            device_type: Null-terminated part number for the connected device.
            device_type_length: the length should be larger than 16
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            device_type_val = create_string_buffer(device_type_length)
            ret = MCM301.mcm301Lib.GetSlotDeviceType(self.hdl, slot_val, byref(device_type_val), device_type_length)
            device_type[0] = device_type_val.value.decode("utf-8", "ignore").rstrip('\x00') \
                .replace("\r\n", "")
        return ret

    def get_software_limit(self, slot, set_software_limit_cw, soft_limit_cw, set_software_limit_ccw, soft_limit_ccw):
        """ Get the saved limit switch parameters for this slot card.
        Args:
            slot: target slot (4,5,6)
            set_software_limit_cw: This indicator indicates whether there are software cw restrictions. 0: not exist; 1: exist
	        soft_limit_cw: the clockwise software limit.
	        set_software_limit_ccw: this indicator indicates whether there are software ccw restrictions. 0: not exist; 1: exist
	        soft_limit_ccw: the clockwise software limit
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            set_software_limit_cw_val = c_int(0)
            soft_limit_cw_val = c_int(0)
            set_software_limit_ccw_val = c_int(0)
            soft_limit_ccw_val = c_int(0)
            ret = MCM301.mcm301Lib.GetSoftwareLimit(self.hdl, slot_val, byref(set_software_limit_cw_val),
                                                    byref(soft_limit_cw_val), byref(set_software_limit_ccw_val),
                                                    byref(soft_limit_ccw_val))
            set_software_limit_cw[0] = set_software_limit_cw_val.value
            soft_limit_cw[0] = soft_limit_cw_val.value
            set_software_limit_ccw[0] = set_software_limit_ccw_val.value
            soft_limit_ccw[0] = soft_limit_ccw_val.value
        return ret

    def get_EFSHW_info(self, info):
        """ Get the hardware info for the MCM - device’s embedded file system.
        Args:
            info:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            info_val = EFSHWInfoStruct()
            ret = MCM301.mcm301Lib.GetEFSHWInfo(self.hdl, byref(info_val))
            info[0] = [info_val.available, info_val.version, info_val.page_size,
                       info_val.pages_supported, info_val.maximum_files, info_val.files_remain, info_val.pages_remain]
        return ret

    def set_soft_limit_value(self, slot, cw_value, ccw_value):
        """ Set the software limit by value.
        Args:
            slot: target slot (4,5,6)
            cw_value: cw encoder limit, default value:2147483647
            ccw_value: ccw encoder limit, default value:-2147483648
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            cw_value_val = c_int(cw_value)
            ccw_value_val = c_int(ccw_value)
            ret = MCM301.mcm301Lib.SetSoftLimitValue(self.hdl, slot_val, cw_value_val, ccw_value_val)
        return ret

    def set_EFSFile_info(self, file_name, file_attribute, file_length):
        """ Set file information from the file system.
        Args:
            file_name: Identifier for the file
            file_attribute: 0x01:APT Read Allowed;
                            0x02:APT Write Allowed;
                            0x04:APT Delete Allowed;
                            0x08:Firmware Read Allowed;
                            0x10:Firmware Write Allowed;
                            0x20:Firmware Delete Allowed
            file_length: Length of the file in pages.When zero, this will delete an existing file. When non-zero,
                         this will create a new file
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            file_name_val = c_char(file_name)
            file_attribute_val = c_byte(file_attribute)
            file_length_cal = c_byte(file_length)
            ret = MCM301.mcm301Lib.SetEFSFileInfo(self.hdl, file_name_val, file_attribute_val, file_length_cal)
        return ret

    def get_EFSFile_info(self, file_name, info):
        """ Get file information from the file system.
        Args:
            file_name: Identifier for the file
            info: the file information
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            file_name_val = c_char(file_name)
            info_val = EFSFileInfoStruct()
            ret = MCM301.mcm301Lib.GetEFSFileInfo(self.hdl, file_name_val, byref(info_val))
            info[0] = [info_val.file_name, info_val.exist, info_val.owned, info_val.attributes, info_val.file_size]
        return ret

    def get_EFSFile_data(self, file_name, file_address, read_length, data_target):
        """ Get the data on a file.
        Args:
            file_name: Identifier for the file
            file_address: The file address to begin reading data.
            read_length: The maximum number of bytes to read.
            data_target:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            file_address_val = c_int(file_address)
            file_name_val = c_char(file_name)
            data_val = (c_byte * read_length)()
            ret = MCM301.mcm301Lib.GetEFSFileData(self.hdl, file_name_val, file_address_val, read_length,
                                                  byref(data_val))
            for i in range(read_length):
                data_target[i] = data_val[i]
        return ret

    def set_EFSFile_data(self, file_name, file_address, data, data_length):
        """ Set the data on a file.
        Args:
            file_name: Identifier for the file
            file_address: The file address to begin reading data.
            data: The data to write to the file
            data_length:
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            file_name_val = c_char(file_name)
            file_address_val = c_int(file_address)
            data_val = (c_byte * data_length)()
            for i in range(data_length):
                data_val[i] = data[i]
            ret = MCM301.mcm301Lib.SetEFSFileData(self.hdl, file_name_val, file_address_val, byref(data_val),
                                                  data_length)
        return ret

    def set_EEPROM_PARAMS_home(self, slot):
        """ Set the home info to EEPROM
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.SetEEPROMPARAMSHome(self.hdl, slot_val)
        return ret

    def set_EEPROM_PARAMS_jog_params(self, slot):
        """ Set the Jog Parameters to EEPROM
        Args:
            slot: target slot (4,5,6)
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            ret = MCM301.mcm301Lib.SetEEPROMPARAMSJogParams(self.hdl, slot_val)
        return ret

    def set_velocity(self, slot, direction, velocity):
        """ Sets the velocity and direction of a slot card.
        Args:
            slot: target slot (4,5,6)
            direction: 0:counter-clockwise. 1:clockwise
            velocity: Percentage of the maximum velocity for the slot card. 0<=velocity<=100
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            direction_val = c_byte(direction)
            velocity_val = c_byte(velocity)
            ret = MCM301.mcm301Lib.SetVelocity(self.hdl, slot_val, direction_val, velocity_val)
        return ret

    def set_home_info(self, slot, home_direction):
        """ Set the homing configuration for the slot card.
        Args:
            slot: target slot (4,5,6)
            home_direction: 0 Home in the clockwise direction.1 Home in the counter - clockwise direction
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            home_direction_val = c_byte(home_direction)
            ret = MCM301.mcm301Lib.SetHomeInfo(self.hdl, slot_val, home_direction_val)
        return ret

    def get_home_info(self, slot, home_direction):
        """  Query the homing configuration for the slot card.
        Args:
            slot: target slot (4,5,6)
            home_direction: 0 Home in the clockwise direction.1 Home in the counter - clockwise direction
        Returns:
            0: Success; negative number: failed.
        """
        ret = -1
        if self.hdl >= 0:
            slot_val = c_byte(slot)
            home_direction_val = c_byte(0)
            ret = MCM301.mcm301Lib.GetHomeInfo(self.hdl, slot_val, byref(home_direction_val))
            home_direction[0] = home_direction_val.value
        return ret
