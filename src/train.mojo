from image import TrainImage
from load_file import read_image_file
from gpu_mem import (
    get_gpu,
    enqueue_create_matrix,
    enqueue_images_to_gpu_matrix,
)
from gpu_ops import forward_propagation
from bit import next_power_of_two
from gpu.id import block_idx

# from gpu_ops import print_matrix, _test_limits

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

    # fn print_image(x: __type_of(x)):
    #     alias px = next_power_of_two(img_pixels)
    #     alias chars = " .,:-=+*#%@"
    #     alias charset_size = len(chars)
    #     tot = Float32()
    #     for i in range(img_pixels):
    #         v = x[i, block_idx.x + 4].reduce_add()
    #         idx = (v / 255.0) * (charset_size - 1)
    #         print(chars[idx.cast[DType.uint8]()], end="")
    #         if i % 28 == 0:
    #             print()
    #     if tot > 0.0:
    #         print(tot)

    # gpu.enqueue_function[print_image](x, grid_dim=1, block_dim=1)

    alias iterations = 500
    # z1, a1, z2, a2 = forward_propagation(gpu, w1, b1, w2, b2, x)
    z1 = forward_propagation(gpu, w1, b1, w2, b2, x)
    # for i in range(iterations):
    #     z1, a1, z2, a2 = forward_propagation(gpu, w1, b1, w2, b2, x)
    # Backward propagation
    # update parameters

    # if i % 10 == 0:
    #     print("Iteration:", i)
    # preditions = get_predictions(a2)
    # print("Accuracy:", get_accuracy(predictions, y))

    gpu.synchronize()
