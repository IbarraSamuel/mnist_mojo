from data_traits import HasData
from bit import next_power_of_two

alias chars = " .,:-=+*#%@"
alias extended_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


fn print_grayscale[
    W: Writer, dtype: DType, //, charset: String = chars
](owned intensity: Scalar[dtype], mut w: W):
    alias charset_size = len(charset)
    pct = intensity.cast[DType.float32]() / 255.0
    index = (pct * (charset_size - 1)).cast[DType.uint8]()
    w.write(charset[index])


alias TrainImage = TrainData[type = DType.uint8]


struct TrainData[rows: Int, cols: Int, type: DType](Writable, HasData):
    alias dtype = type
    alias size = cols * rows
    alias simd_size = next_power_of_two(Self.size)
    alias Data = SIMD[Self.dtype, Self.simd_size]

    var label: UInt8
    var data: Self.Data

    fn __init__(
        out self,
        data: String,
        init: Int,
        end: Int,
    ) raises:
        # Slice the span
        string = data[init:end]
        # string = String(buffer=List(self_data))
        init_pos = string.find(",")
        self.label = Int(string[:init_pos])
        elems = (string^)[init_pos + 1 :].split(",")
        final_data = Self.Data(0)
        # nums_list = List[Float32]()
        for i in range(len(elems)):
            final_data[i] = Int(elems[i])

        self.data = final_data

    fn __copyinit__(out self, other: Self):
        self.label = other.label
        self.data = other.data

    fn __moveinit__(out self, owned other: Self):
        self.label = other.label
        self.data = other.data

    fn get_data(self) -> Self.Data:
        return self.data

    fn write_to[w: Writer](self, mut writer: w):
        for i in range(Self.size):
            print_grayscale[extended_chars](self.data[i], writer)
            if i % cols == 0:
                writer.write("\n")
        writer.write("\n")

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, x: Int, y: Int) -> Scalar[Self.dtype]:
        return self.data[x + y * cols]

    fn __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self.data[i]
