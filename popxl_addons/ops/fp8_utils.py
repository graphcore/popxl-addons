# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from collections import namedtuple

# Define each floating point format that we use
FpDef = namedtuple(
    "FpDef",
    [
        "num_exp_bits",  # Number of exponent bits
        "num_mantissa_bits",  # Number of mantissa bits
        "fix_exponent_bias",  # The bias value to add to the exponent
        "include_denorm",  # Do we consider denormalised values in the format?
        "has_nans",  # True if a whole exponent "code" is used to represent Nan
        "max_value",  # Maximum value this format can represent
        "max_exp",  # Maximum exponent
        "min_exp",  # Min exponent
    ],
)

ieee_fp32_no_denorms = FpDef(8, 23, 127, False, True, 3.4028235e38, 127, -126)
ieee_fp16_denorms = FpDef(5, 10, 15, True, True, 65504, 15, -14)
float8_152_def = FpDef(5, 2, 16, True, False, 57344, 15, -17)
float8_143_def = FpDef(4, 3, 8, True, False, 240, 7, -10)
