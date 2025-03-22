from image import TrainImage
from load_file import read_image_file
from gpu_mem import (
    get_gpu,
    enqueue_build_matrix,
    enqueue_images_to_gpu_matrix,
)

alias img_rows = 28
alias img_cols = 28


alias Image = TrainImage[img_rows, img_cols]

alias train_filepath = "digit-recognizer/train.csv"
alias test_filepath = "digit-recognizer/test.csv"

alias train_size = 42000
alias dtype = DType.uint8


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
    w1b, w1 = enqueue_build_matrix[10, img_rows * img_cols, dtype=dtype](gpu)
    b1b, b1 = enqueue_build_matrix[10, 1, dtype=dtype](gpu)
    w2b, w2 = enqueue_build_matrix[10, 10, dtype=dtype](gpu)
    b2b, b2 = enqueue_build_matrix[10, 1, dtype=dtype](gpu)

    print("load train to gpu")
    gpu.synchronize()
    xb, x = enqueue_build_matrix[img_rows * img_cols, train_size, dtype=dtype](
        gpu
    )
    enqueue_images_to_gpu_matrix(gpu, xb, x, images)
    gpu.synchronize()
