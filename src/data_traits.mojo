from std.bit import next_power_of_two


trait HasData:
    comptime dtype: DType
    comptime size: Int

    def get_data(self) -> SIMD[Self.dtype, next_power_of_two(Self.size)]:
        ...


trait HasLabel:
    def get_label(self) -> Int:
        ...


comptime Loadable = HasData & HasLabel
