from bit import next_power_of_two


trait HasData(CollectionElement):
    alias dtype: DType
    alias size: Int

    fn get_data(self) -> SIMD[dtype, next_power_of_two(Self.size)]:
        ...


trait HasLabel(CollectionElement):
    fn get_label(self) -> Int:
        ...


trait Loadable(HasData, HasLabel):
    ...
