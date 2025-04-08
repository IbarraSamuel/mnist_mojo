from pathlib import Path
from algorithm import sync_parallelize
import os

from image import TrainData


fn arange_data[
    dtype: DType, rows: Int, cols: Int, size: Int = -1
](train_data_string: String) raises -> List[TrainData[rows, cols, dtype]]:
    alias TD = TrainData[rows, cols, dtype]
    jumps = List[Int]()
    n = 0
    while True:
        next_jump = train_data_string.find("\n", n)
        if next_jump == -1:
            break

        jumps.append(next_jump)
        n = next_jump + 1

    # All jumps represents a newline
    file_rows = len(jumps)
    if size != -1 and file_rows != size:
        msg = "Wrong sizes, we found {} newlines, but the size should be {}"
        err = msg.as_string_slice().format(len(jumps), size)
        os.abort(err)

    frst = TD(data=train_data_string, init=0, end=jumps[0])
    images = List[TD]()

    # Fill with something
    for _ in range(file_rows):
        images.append(frst)

    # Then, fill it with multiple threads
    @parameter
    fn calc_each(i: Int) raises:
        img = TD(
            data=train_data_string,
            init=jumps[i],
            end=jumps[i + 1],
        )
        images[i + 1] = img

    sync_parallelize[calc_each](len(jumps) - 1)

    return images


fn read_image_file[
    path: String,
    *,
    rows: Int,
    cols: Int,
    dtype: DType,
    size: Int = -1,
]() raises -> List[TrainData[rows, cols, dtype]]:
    file = Path(path)
    text_data = file.read_text()

    # Ignore the headers
    text_data = text_data[text_data.find("\n") + 1 :]

    return arange_data[dtype, rows, cols, size](text_data)
