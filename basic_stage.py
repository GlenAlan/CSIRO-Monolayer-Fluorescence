from MCM301_COMMAND_LIB import *
import time

print("*** MCM301 device python example ***")
mcm301obj = MCM301()

devs = MCM301.list_devices()
print(devs)
if len(devs) <= 0:
    print('There is no devices connected')
    exit()
device_info = devs[0]
sn = device_info[0]
print("connect ", sn)
hdl = mcm301obj.open(sn, 115200, 3)
if hdl < 0:
    print("open ", sn, " failed. hdl is ", hdl)
    exit()
if mcm301obj.is_open(sn) == 0:
    print("MCM301IsOpen failed")
    mcm301obj.close()
    exit()
input()