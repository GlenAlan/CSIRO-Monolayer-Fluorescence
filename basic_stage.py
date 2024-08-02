from MCM301_COMMAND_LIB import *
import time
import random

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

step_size = 5000
for stage_num in [4, 5, 6]:
    result = mcm301obj.set_jog_params(stage_num, step_size)
    if result < 0:
        print("set_jog_params failed")
    else:
        print("set_jog_params:", step_size)

for i in range(20):
    stage_num, stage_direction = random.randint(4, 6), random.randint(0, 1)
    mcm301obj.move_jog(stage_num, stage_direction)
    print(f"Moivng stage {stage_num}, {'clockwise'*stage_direction + 'counter-clockwise'*(1-stage_direction)}")
    time.sleep(1)
