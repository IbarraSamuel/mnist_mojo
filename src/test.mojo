fn main():
    alias sum_size = 10000
    print("Sum range: [ 0,", sum_size, ")")
    expected = sum_size * (sum_size + 1) // 2 - sum_size
    print("Value expected:", expected)

    cumulative_sum[
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    ](sum_size, expected)


fn cumulative_sum[*types: DType](to: Int, expected: Int):
    alias size = len(VariadicList(types))

    @parameter
    for i in range(size):
        alias dt = types[i]
        var v = Scalar[dt](0)
        print("summing dtype", dt, "for", to, "elements")
        print("-- max for dtype", dt, ":", v.MAX_FINITE)
        for i in range(to):
            v += Scalar[dt](i)
        print("-- result for", dt, "addition from 0..10000:", v)
        print("-- match expected?:", v == Scalar[dt](expected))
