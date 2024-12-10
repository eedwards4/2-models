# Created by Ethan Edwards on 12/6/2024

# Imports
import numpy as np
import pandas as pd

import perceptron
import neural_net


def open_images(path):
    """
    Opens the csv files and extracts the images from them and returns them
    """
    images = []
    data = pd.read_csv(path)
    headers = data.columns.values

    labels = data[headers[0]]
    labels = labels.values.tolist()

    pixels = data.drop(headers[0], axis=1)

    for i in range(0, data.shape[0]):
        row = pixels.iloc[i].to_numpy()
        grid = np.reshape(row, (28, 28))
        images.append(grid)
    return labels, images


def get_BW(image):
    """
    Converts the image to black and white
    """
    pixels = []
    for x in range(28):
        for y in range(28):
            if (int(image[x][y]) > 128):
                pixels.append(1)
            else:
                pixels.append(0)
    return np.reshape(pixels, (28, 28))


def vert_inters(image):
    """
    Gets the number of vertical intersections in black and white image
    """
    counts = []
    prev = 0
    for y in range(28):
        count = 0
        for x in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts)/28
    maximum = max(counts)
    return average, maximum


def horiz_inters(image):
    """
    Gets the number of horizontal intersections in black and white image
    """
    counts = []
    for x in range(28):
        count = 0
        prev = 0
        for y in range(28):
            current = int(image[x][y])
            if (prev != current):
                count += 1
            prev = current
        counts.append(count)
    average = sum(counts)/28
    maximum = max(counts)
    return average, maximum


def calc_density(image):
    """
    Calculates the density
    """
    count = 0
    for x in range(28):
        for y in range(28):
            count = count + int(image[x][y])
    return count / (28 * 28)


def calc_symmetry(image):
    """
    Calculates the degree of symmetry
    """
    image = np.array(image)
    reflected_image = np.fliplr(image)
    xor_result = np.bitwise_xor(image, reflected_image)
    symmetry_measure = np.mean(xor_result)
    return symmetry_measure


def centroid_distance_variance(image):
    """
    Calculate the variance of distances from the centroid to the "on" pixels.
    """
    image = np.array(image)
    on_pixels = np.argwhere(image > 0)  # Get indices of all "on" pixels
    centroid = np.mean(on_pixels, axis=0)  # Compute centroid
    distances = np.linalg.norm(on_pixels - centroid, axis=1)  # Euclidean distances
    return np.var(distances)


def aspect_ratio(image):
    """
    Calculate the aspect ratio of the bounding box surrounding the digit.
    """
    binary_img = (np.array(image) > 0).astype(np.uint8)
    coords = np.column_stack(np.where(binary_img > 0))  # Get coordinates of non-zero pixels
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    return height / width if width > 0 else 0  # Avoid division by zero


def calc_euler(image):
    """
    Calculation of the Euler number for a binary image.
    """
    # Ensure the image is binary
    binary_image = (image > 0).astype(int)

    # Pad the image to handle edges
    padded_image = np.pad(binary_image, pad_width=1, mode='constant', constant_values=0)

    # Extract 2x2 blocks using slicing
    block_1 = padded_image[:-1, :-1]  # Top-left
    block_2 = padded_image[:-1, 1:]  # Top-right
    block_3 = padded_image[1:, :-1]  # Bottom-left
    block_4 = padded_image[1:, 1:]  # Bottom-right

    # Sum pixel values in each 2x2 block
    block_sums = block_1 + block_2 + block_3 + block_4

    # Count occurrences of 1s and 3s in the blocks
    n1 = np.sum(block_sums == 1)  # Single foreground pixel
    n3 = np.sum(block_sums == 3)  # Three foreground pixels

    # Compute Euler number
    euler_number = (n1 - n3) // 4
    return euler_number


def diag_symmetry(image):
    """
    Calculates the diagonal symmetry of a 2D grayscale image.
    """
    image = np.array(image)
    # Flip vertically and transpose
    flipped_image = np.flipud(image).T

    # Compute pixel-wise absolute difference between the image and its flipped version
    diff = np.abs(image - flipped_image)

    # Calculate the average symmetry score
    symmetry_score = 1 - (np.sum(diff) / (255 * image.size))  # Normalize to range [0, 1]

    return symmetry_score


def normalizer(features):
    """
    Normalize the features.
    """
    normalized = []

    features = np.array(features)

    data = features[:, :-1]
    labels = features[:, -1]

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    normalized_data = (data - means) / stds

    # Combine the normalized features with the labels
    normalized = np.hstack((normalized_data, labels.reshape(-1, 1)))

    return normalized.tolist()


def feature_extractor(images, labels):
    """
    Extract features from the images.
    """
    features = []
    for image in images:
        BW = get_BW(image)
        vert_avg, vert_max = vert_inters(BW)
        horiz_avg, horiz_max = horiz_inters(BW)
        density = calc_density(BW)
        symmetry = calc_symmetry(BW)
        centroid_variance = centroid_distance_variance(BW)
        aspect = aspect_ratio(BW)
        euler = calc_euler(BW)
        diag_sym = diag_symmetry(BW)

        feature_row = ([
            vert_avg, vert_max, horiz_avg, horiz_max, density, symmetry,
            centroid_variance, aspect, euler, diag_sym
        ])

        feature_row.append(labels.pop(0))

        features.append(feature_row)

    return features


def dataloader_raw(filename):
    """
    Load the data from the file.
    """
    labels, images = open_images(filename)
    return images, labels


def dataloader(filename):
    """
    Load the data from the file.
    """
    labels, images = open_images(filename)
    features = feature_extractor(images, labels)
    return features


def main():
    """
    Main function
    """

    # Compile-time flags/options
    skip_1 = False
    skip_2 = False
    skip_3 = False
    train_dir = "./dataset/train"
    val_dir = "./dataset/val"
    test_dir = "./dataset"

    # Data
    TRAIN = []
    VAL = []

    print("Starting...")
    if not skip_1:
        print("Loading data for part 1...")
        TRAIN += dataloader(f"{train_dir}/train7.csv")
        TRAIN += dataloader(f"{train_dir}/train9.csv")
        VAL += dataloader(f"{val_dir}/valid7.csv")
        VAL += dataloader(f"{val_dir}/valid9.csv")
        TRAIN = normalizer(TRAIN)
        VAL = normalizer(VAL)
        # Scramble the data
        np.random.shuffle(TRAIN)
        np.random.shuffle(VAL)
        print("Part 1 data loaded, starting training run...")
        model = perceptron.perceptron(2, 10, 0.01)
        model.train(TRAIN, VAL, 1000, 2)
        print("Training complete. Testing on test data...")
        TEST = dataloader(f"{val_dir}/valid7.csv")
        TEST += dataloader(f"{val_dir}/valid9.csv")
        TEST = normalizer(TEST)
        np.random.shuffle(TEST)

        model.eval(TEST)

        print("Part 1 complete.")

    if not skip_2:
        print("\nLoading data for part 2...")
        TRAIN = []
        VAL = []
        for i in range(0, 10):
            TRAIN += dataloader(f"{train_dir}/train{i}.csv")
            VAL += dataloader(f"{val_dir}/valid{i}.csv")

        TRAIN = normalizer(TRAIN)
        VAL = normalizer(VAL)

        np.random.shuffle(TRAIN)
        np.random.shuffle(VAL)

        print("Part 2 data loaded, starting training run...")
        model = perceptron.perceptron(10, 10, 0.01)
        model.train(TRAIN, VAL, 1000, 10)
        print("Training complete. Testing on test data...")
        TEST = []
        TEST += dataloader(f"{test_dir}/test1.csv")
        TEST = normalizer(TEST)

        model.eval(TEST)
        print("Part 2 complete.")

    if not skip_3:
        print("\nLoading data for part 3...")
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for i in range(0, 10):
            x_tmp, y_tmp = dataloader_raw(f"{train_dir}/train{i}.csv")
            x_train += x_tmp
            y_train += y_tmp
            x_tmp, y_tmp = dataloader_raw(f"{val_dir}/valid{i}.csv")
            x_val += x_tmp
            y_val += y_tmp

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten training data
        x_val = x_val.reshape(x_val.shape[0], -1)  # Flatten validation data

        print("Part 3 data loaded, starting training run...")

        model = neural_net.neural_net(input_size=784, hidden_size=100, output_size=10, learning_rate=0.01)

        model.train(x_train, y_train, x_val, y_val)

        print("Training complete. Testing on test data...")

        x_test, y_test = dataloader_raw(f"{test_dir}/test1.csv")

        x_test = np.array(x_test)

        x_test = x_test.reshape(x_test.shape[0], -1)

        model.validate(x_test, y_test)
        print("Part 3 complete.")


if __name__ == "__main__":
    main()
