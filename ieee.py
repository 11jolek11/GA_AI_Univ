import math


def ieee_754_conversion(n, sgn_len=1, exp_len=8, mant_len=23):
    """
    Converts an arbitrary precision Floating Point number.
    Note: Since the calculations made by python inherently use floats, the accuracy is poor at high precision.

    :param n: An unsigned integer of length `sgn_len` + `exp_len` + `mant_len` to be decoded as a float
    :param sgn_len: number of sign bits
    :param exp_len: number of exponent bits
    :param mant_len: number of mantissa bits
    :return: IEEE 754 Floating Point representation of the number `n`
    """
    if n >= 2 ** (sgn_len + exp_len + mant_len):
        raise ValueError("Number n is longer than prescribed parameters allows")

    sign = (n & (2 ** sgn_len - 1) * (2 ** (exp_len + mant_len))) >> (exp_len + mant_len)
    exponent_raw = (n & ((2 ** exp_len - 1) * (2 ** mant_len))) >> mant_len
    mantissa = n & (2 ** mant_len - 1)

    sign_mult = 1
    if sign == 1:
        sign_mult = -1

    if exponent_raw == 2 ** exp_len - 1:  # Could be Inf or NaN
        if mantissa == 2 ** mant_len - 1:
            return float('nan')  # NaN

        return sign_mult * float('inf')  # Inf

    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)

    if exponent_raw == 0:
        mant_mult = 0  # Gradual Underflow
    else:
        mant_mult = 1

    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult


def reverse_ieee_754_conversion(f, sgn_len=1, exp_len=8, mant_len=23):
    """
    Converts an IEEE 754 Floating Point representation to a precision Floating Point number.
    :param f: IEEE 754 Floating Point representation of a number
    :param sgn_len: number of sign bits
    :param exp_len: number of exponent bits
    :param mant_len: number of mantissa bits
    :return: precision Floating Point number
    """
    sign = 0 if f >= 0 else 1
    sign_bits = sign * (2 ** (exp_len + mant_len))

    if abs(f) == float('inf'):
        exponent_bits = 2 ** exp_len - 1
        mantissa_bits = 0 if f >= 0 else 2 ** mant_len - 1
    elif abs(f) == float('nan'):
        exponent_bits = 2 ** exp_len - 1
        mantissa_bits = 2 ** mant_len - 1
    else:
        exponent = 0
        mantissa = 0

        if f != 0:
            exponent = int(math.log(abs(f), 2))
            mantissa = abs(f) / (2 ** exponent) - 1

        exponent += (2 ** (exp_len - 1) - 1)
        exponent_bits = exponent
        mantissa_bits = int(mantissa * (2 ** mant_len))

    n = sign_bits | (exponent_bits << mant_len) | mantissa_bits
    return n

def ieee745_32(N): # ieee-745 bits (max 32 bit)
    a = int(N[0])        # sign,     1 bit
    b = int(N[1:9],2)    # exponent, 8 bits
    c = int("1"+N[9:], 2)# fraction, len(N)-9 bits

    return (-1)**a * c /( 1<<( len(N)-9 - (b-127) ))

def ieee745_64(N): # ieee-745 bits (max 32 bit)
    a = int(N[0])        # sign,     1 bit
    b = int(N[1:12],2)    # exponent, 8 bits
    c = int("1"+N[12:], 2)# fraction, len(N)-9 bits

    return (-1)**a * c /( 1<<( len(N)-12 - (b-1023) ))

def ieee745_x(N, sgn_len=1, exp_len=3, mant_len=60): # ieee-745 bits (max 32 bit)
    assert(len(N) == sgn_len + exp_len + mant_len)
    a = int(N[sgn_len-1])        # sign,     1 bit
    b = int(N[sgn_len:exp_len+1],2)    # exponent, 8 bits
    c = int("1"+N[exp_len+1:], 2)# fraction, len(N)-9 bits

    return (-1)**a * c /( 1<<( len(N)- exp_len+1 - (b- (2**exp_len)) -1))

if __name__ == '__main__':
    import struct
    m = struct.unpack('I', struct.pack('f', -7.125))[0]
    print("Unsigned Int Recieved:", str(m))
    print("Manually converted:", ieee_754_conversion(3236167680, exp_len=8, mant_len=23))
    # print(ieee745_32("110000011010010011"))
    print(ieee745_32("10111110000000000000000000000000"))
    print(ieee745_64("1111111111110000000000000000000000000000000000000000000000000000"))
    # print(ieee745_x("0110111111000000000000000000000000000000000000000000000000000000"))
