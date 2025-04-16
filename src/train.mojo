from image import TrainImage
from load_file import read_image_file
from gpu_mem import (
    get_gpu,
    enqueue_create_matrix,
    enqueue_create_labels,
    enqueue_images_to_gpu_matrix,
    Layout,
)
from gpu_ops import (
    forward_propagation,
    backward_propagation,
    one_hot_y,
    update_parameters,
    get_predictions,
    print_accuracy,
)
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
    alias max_y = 9
    alias ldim = max_y + 1
    _, w1 = enqueue_create_matrix[Layout(ldim, img_pixels), dtype, True](gpu)
    _, b1 = enqueue_create_matrix[size=ldim, dtype=dtype, randomize=True](gpu)
    _, w2 = enqueue_create_matrix[Layout(ldim, ldim), dtype, True](gpu)
    _, b2 = enqueue_create_matrix[size=ldim, dtype=dtype, randomize=True](gpu)

    print("load train to gpu")
    xb, x = enqueue_create_matrix[Layout(img_pixels, train_size), dtype](gpu)
    enqueue_images_to_gpu_matrix(gpu, xb, x, images)

    yb, y = enqueue_create_matrix[Layout(train_size), dtype](gpu)
    enqueue_create_labels(gpu, yb, y, images)

    # This should match to the max_y + 1 == 10 -> the dimention
    hot_y = one_hot_y[max_y=9](gpu, y)

    alias iterations = 100
    alias alpha = Scalar[dtype](0.001)

    for i in range(iterations):
        z1, a1, _, a2 = forward_propagation(gpu, x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_propagation(gpu, x, z1, a1, a2, w2, hot_y)
        w1, b1, w2, b2 = update_parameters[alpha](
            gpu, w1, b1, w2, b2, dw1, db1, dw2, db2
        )

        if i % 10 == 0:
            print("Iteration:", i)
            var predictions = get_predictions(gpu, a2)
            print_accuracy(gpu, predictions, y)

    gpu.synchronize()
