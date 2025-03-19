from pathlib import Path

# from gpu.host import DeviceContext, DeviceBuffer
# from layout import Layout, LayoutTensor
# from gpu import thread_idx, block_idx, barrier
from memory import Span

from tensor import Tensor
import os

from algorithm import sync_parallelize

alias file_rows = 42000
alias img_rows = 28
alias img_cols = 28
alias img_pixels = img_rows * img_cols

alias num_offset = ord("0")
alias linebreak_byte: UInt8 = ord("\n")
alias chars = " .,:-=+*#%@"
alias extended_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
alias grads = List[String](" ", "░", "▒", "▓", "█")


fn print_grayscale[
    W: Writer, //, charset: List[String] = grads
](owned intensity: UInt8, mut w: W):
    alias charset_size = len(charset)
    pct = intensity.cast[DType.float32]() / 255.0
    index = (pct * (charset_size - 1)).cast[DType.uint8]()
    w.write(charset[index])


fn print_grayscale[
    W: Writer, //, charset: String = chars
](owned intensity: UInt8, mut w: W):
    alias charset_size = len(charset)
    pct = intensity.cast[DType.float32]() / 255.0
    index = (pct * (charset_size - 1)).cast[DType.uint8]()
    w.write(charset[index])


fn print_grayscale[W: Writer, //](owned intensity: UInt8, mut w: W):
    gray_code = 232 + Int(intensity.cast[DType.float32]() * 23.0 / 255.0)
    w.write("\033[48;5;", gray_code, "m")
    # alias charset_size = len(charset)
    # pct = intensity.cast[DType.float32]() / 255.0
    # index = (pct * (charset_size - 1)).cast[DType.uint8]()
    # w.write(charset[index])


struct TrainImage(Writable):
    var label: UInt8
    var data: List[UInt8]

    fn __init__(
        out self,
        data: Span[UInt8],
        init: Int,
        end: Int,
    ) raises:
        # Slice the span
        self_data = data[init:end]
        string = String(buffer=List(self_data))
        init_pos = string.find(",")
        self.label = Int(string[:init_pos])
        elems = (string^)[init_pos + 1 :].split(",")
        nums_list = List[UInt8]()
        for elem in elems:
            nums_list.append(Int(elem[]))

        self.data = nums_list

    fn __copyinit__(out self, other: Self):
        self.label = other.label
        self.data = other.data

    fn __moveinit__(out self, owned other: Self):
        self.label = other.label
        self.data = other.data

    fn write_to[w: Writer](self, mut writer: w):
        for i in range(img_pixels):
            print_grayscale[extended_chars](self.data[i], writer)
            if i % img_cols == 0:
                writer.write("\n")
        writer.write("\n")

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, x: Int, y: Int) -> ref [self.data] UInt8:
        return self.data[x + y * img_cols]

    fn __getitem__(self, i: Int) -> ref [self.data] UInt8:
        return self.data[i]


fn main() raises:
    var data_folder = Path("digit-recognizer")
    # test = data_folder / "test.csv"
    train = data_folder / "train.csv"

    # test_data = test.read_bytes()
    train_data_string = train.read_text()
    # Other way to find "\n" in a text?
    jumps = List[Int]()
    n = 0
    while n < len(train_data_string):
        next_jump = train_data_string.find("\n", n)
        if next_jump == -1:
            break

        jumps.append(next_jump)
        n = next_jump + 1

    print("Building images...")

    images = List[TrainImage]()
    for i in range(len(jumps) - 1):
        img = TrainImage(
            data=train_data_string.as_bytes(),
            init=jumps[i],
            end=jumps[i + 1],
        )
        images.append(img)

    # To show it up
    # for i in range(len(images)):
    #     print("image:", i)
    #     print(images[i])

    # for v in img.data:
    #     print(Int(v[]), end=", ")

    # print("Create inline array...")
    # train_data = InlineArray[TrainImage, file_rows, run_destructors=True](
    #     fill=img
    # )

    # @parameter
    # fn parse_line(i: Int) raises:
    #     start = jumps[i]
    #     end = jumps[i + 1]

    #     curr_data = train_data_string.as_bytes()
    #     timage = TrainImage(data=curr_data, init=start, end=end)
    #     train_data[i] = timage

    # print("Filling inline array in a parallel way...")
    # sync_parallelize[parse_line](len(jumps) - 1)

    # print("Seeing results...")
    # for i in range(train_data.size):
    #     print(train_data[i])
