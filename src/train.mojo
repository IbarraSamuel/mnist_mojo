from image import TrainImage
from load_file import read_image_file
from gpu_mem import (
    get_gpu,
    enqueue_create_matrix,
    enqueue_create_labels,
    enqueue_images_to_gpu_matrix,
    Layout,
)
from gpu_ops import forward_propagation, backward_propagation, one_hot_y
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

    alias w1_layout = Layout(10, img_pixels)
    _, w1 = enqueue_create_matrix[Layout(10, img_pixels), dtype, True](gpu)
    _, b1 = enqueue_create_matrix[Layout(10, 1), dtype, randomize=True](gpu)
    _, w2 = enqueue_create_matrix[Layout(10, 10), dtype, randomize=True](gpu)
    _, b2 = enqueue_create_matrix[Layout(10, 1), dtype, randomize=True](gpu)

    print("load train to gpu")
    xb, x = enqueue_create_matrix[Layout(img_pixels, train_size), dtype](gpu)
    enqueue_images_to_gpu_matrix(gpu, xb, x, images)

    yb, y = enqueue_create_matrix[Layout(train_size), dtype](gpu)
    enqueue_create_labels(gpu, yb, y, images)

    hot_y = one_hot_y(gpu, y)

    alias iterations = 10

    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(gpu, x, w1, b1, w2, b2)

        # Backward propagation
        dw1, db1, dw2, db2 = backward_propagation(gpu, x, z1, a1, a2, w2, hot_y)

        # update parameters

        if i % 10 == 0:
            print("Iteration:", i)
    # preditions = get_predictions(a2)
    # print("Accuracy:", get_accuracy(predictions, y))

    gpu.synchronize()
