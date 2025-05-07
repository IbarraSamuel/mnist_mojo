from bit import next_power_of_two


trait HasData:
    alias dtype: DType
    alias size: Int

    fn get_data(self) -> SIMD[dtype, next_power_of_two(Self.size)]:
        ...


trait HasLabel:
    fn get_label(self) -> Int:
        ...


alias Loadable = HasData & HasLabel
