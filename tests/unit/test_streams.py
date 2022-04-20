# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl

import popxl_addons as addons


def test_input_streams():
    ir = popxl.Ir()
    with ir.main_graph:
        inputs = addons.InputStreams(x=((1, 4), popxl.float32), y=((4, ), popxl.int32))

    assert isinstance(inputs.x, popxl.HostToDeviceStream)
    assert isinstance(inputs.y, popxl.HostToDeviceStream)

    assert inputs[0] == inputs.x
    assert inputs[1] == inputs.y

    for stream in inputs:
        assert isinstance(stream, popxl.HostToDeviceStream)


def test_output_streams():
    ir = popxl.Ir()
    with ir.main_graph:
        outputs = addons.OutputStreams(x=((1, 4), popxl.float32), y=((4, ), popxl.int32))

    assert isinstance(outputs.x, popxl.DeviceToHostStream)
    assert isinstance(outputs.y, popxl.DeviceToHostStream)

    assert outputs[0] == outputs.x
    assert outputs[1] == outputs.y

    for stream in outputs:
        assert isinstance(stream, popxl.DeviceToHostStream)
