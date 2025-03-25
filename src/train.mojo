from image import TrainImage
from load_file import read_image_file
from gpu_mem import (
    get_gpu,
    enqueue_create_matrix,
    enqueue_images_to_gpu_matrix,
)
from gpu_ops import print_matrix

alias img_rows = 28
alias img_cols = 28
alias img_pixels = img_rows * img_cols

alias Image = TrainImage[img_rows, img_cols]

alias train_filepath = "digit-recognizer/train.csv"
alias test_filepath = "digit-recognizer/test.csv"

alias train_size = 42000
alias dtype = DType.float32


fn main() raises:
    print("Reading files...")
    var images = read_image_file[
        train_filepath,
        size=train_size,
        rows=img_rows,
        cols=img_cols,
        dtype=dtype,
    ]()

    print("Files readed!")
    # for img in range(len(images)):
    #     print(images[img])
    # print("All images readed!")

    gpu = get_gpu()

    w1b, w1 = enqueue_create_matrix[
        10, img_pixels, dtype=dtype, randomize=True
    ](gpu)

    b1b, b1 = enqueue_create_matrix[10, 1, dtype=dtype, randomize=True](gpu)
    w2b, w2 = enqueue_create_matrix[10, 10, dtype=dtype, randomize=True](gpu)
    b2b, b2 = enqueue_create_matrix[10, 1, dtype=dtype, randomize=True](gpu)

    print("load train to gpu")
    xb, x = enqueue_create_matrix[img_pixels, train_size, dtype=dtype](gpu)
    enqueue_images_to_gpu_matrix(gpu, xb, x, images)
    gpu.synchronize()

    print_matrix(gpu, b2b, b2)
