"""
Single Precision/FP32. See https://en.wikipedia.org/wiki/Single-precision_floating-point_format

Sign bit: 1 bit
Exponent: 8 bits
    8-bit unsigned in in biased form, i.e. 2^N-127.
    Exponent 0 and 255 are reserved for special numbers, thus has range [1, 254] - 127 = [-126, 127]

Mantissa (or significant precision): 23 bits
    24 bits (23 bits exstored) with implicit leading bit with value 1.
    23-bit unsigned in normalized form, i.e. 1.xxxxxx
    1 is implicit, so it is not stored 
"""
import torch

exponent_range = (1 - 127, 254 - 127)
print(f"{exponent_range[0]:b}->{exponent_range[0]} {exponent_range[1]:b}->{exponent_range[1]}")

mantissa_range = (1, 2 - 2**-23)
print(f"{mantissa_range[0]} {mantissa_range[1]}")

print("For normalized numbers")
largest = 2 ** exponent_range[1] * mantissa_range[1]
smallest = -(2 ** exponent_range[1]) * mantissa_range[1]
smallest_positive = 2 ** exponent_range[0] * mantissa_range[0]
esp = 2**-23
print(f"{largest=} {smallest=} {smallest_positive=} {esp=}")

t = torch.finfo(torch.float32)
assert t.max == largest
assert t.min == smallest
assert t.smallest_normal == smallest_positive
assert t.eps == esp


"""
From ChatGPT
Exponent Value 0:
    Zero: If the exponent is 0 and the mantissa (or significand) is also all zeros, the value represents zero
    Subnormal (Denormal) Numbers: If the exponent is 0 but the mantissa is not all zeros, the value represents
        a subnormal number. In these numbers, there's no implicit leading 1 in the mantissa as in normal numbers
        , and they are used to represent numbers that are too small to be normalized (i.e., numbers smaller 
        than 2^-126)
    Subnormal (or Denormal) numbers are more subject to rounding errors than normalized numbers.

Exponent Value 255:
    Infinity: If the exponent is 255 and the mantissa is all zeros, the value represents infinity. The sign bit
        determines if it's positive or negative infinity.
    NaN (Not a Number): If the exponent is 255 and the mantissa is not all zeros, the value represents NaN.

"""
import struct
import math

def binary_to_float(binary):
    return struct.unpack('f', struct.pack('I', int(binary)))[0]

binary_literal = 0b00000000000000000000000000000000
float_representation = binary_to_float(binary_literal)
assert float_representation == 0.0

binary_literal = 0b00000000000000000000000000000001
float_representation = binary_to_float(binary_literal)
assert float_representation == 2**(-126+-23)

binary_literal = 0b01111111100000000000000000000000
float_representation = binary_to_float(binary_literal)
assert float_representation == float('inf')
assert math.isinf(float_representation)

binary_literal = 0b01111111100000000000000000000001
float_representation = binary_to_float(binary_literal)
assert float_representation != float_representation
assert math.isnan(float_representation)
