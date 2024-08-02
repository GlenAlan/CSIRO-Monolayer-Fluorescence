try:
    from MCM301_COMMAND_LIB import *
    import time
except OSError as ex:
    print("Warning:", ex)


# ------------ Example Device Read&Write for slot 4-------------- #
def device_read_write_demo(mcm301obj):
    print("*** Device Read&Write example")

    border_status_struct = [0]
    result = mcm301obj.get_board_status(border_status_struct)
    if result < 0:
        print("border_status_struct failed", result)
    else:
        print("border_status_struct: ", border_status_struct)

    dim = 20
    result = mcm301obj.set_system_dim(dim)
    if result < 0:
        print("set_system_dim failed", result)
    else:
        print("set system_dim: ", dim)

    dim = [0]
    result = mcm301obj.get_system_dim(dim)
    if result < 0:
        print("get_system_dim failed", result)
    else:
        print("get system_dim: ", dim)

    encoder_counter = 20
    nm = [0]
    result = mcm301obj.convert_encoder_to_nm(4, encoder_counter, nm)
    if result < 0:
        print("convert_encoder_to_nm failed", result)
    else:
        print("convert 20 encoder to nm is ", nm)

    title = b"X-Axis"
    title_len = len(title)
    result = mcm301obj.set_slot_title(4, title, title_len)
    if result < 0:
        print("set_slot_title failed", result)
    else:
        print("set_slot_title to ", title)

    title = [0]
    result = mcm301obj.get_slot_title(4, title, 100)
    if result < 0:
        print("get_slot_title failed")
    else:
        print("get slot title:", title)


# ------------ Example Stage Read&Write for slot 4-------------- #
def stage_read_write_demo(mcm301obj):
    print("*** stage Read&Write example")

    stage_params_info = [0]
    result = mcm301obj.get_stage_params(5, stage_params_info)
    if result < 0:
        print("get_stage_params failed")
    else:
        print("get_stage_params:", stage_params_info)

    state = 1
    result = mcm301obj.set_chan_enable_state(5, state)
    if result < 0:
        print("set_chan_enable_state failed")
    else:
        print("set_chan_enable_state:", state)

    step_size = 10000
    result = mcm301obj.set_jog_params(5, step_size)
    if result < 0:
        print("set_jog_params failed")
    else:
        print("set_jog_params:", step_size)

    velocity = 50
    direction = 0  # direction: 0:counter-clockwise. 1:clockwise
    result = mcm301obj.set_velocity(5, direction, velocity)
    if result < 0:
        print("set_velocity failed")
    else:
        print("set_velocity:", velocity)

    direction = 0  # direction: 0 Counter-Clockwise;1 Clockwise
    result = mcm301obj.move_jog(5, direction)
    if result < 0:
        print("move_jog failed")
    else:
        print("move_jog succeeded")
        time.sleep(5)

    target_encoder = 10000
    result = mcm301obj.move_absolute(5, target_encoder)
    if result < 0:
        print("move_absolute failed")
    else:
        print("move_absolute succeeded")
        time.sleep(5)


def main():
    print("*** MCM301 device python example ***")
    mcm301obj = MCM301()
    try:
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

        device_read_write_demo(mcm301obj)
        print("---------------------------Device Read&Write finished-------------------------")
        stage_read_write_demo(mcm301obj)
        print("---------------------------Stage Read&Write finished-------------------------")
        mcm301obj.close()

    except Exception as e:
        print("Warning:", e)
    print("*** End ***")
    input()


main()
