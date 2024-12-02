import os
import random
import time


def generate_imglist(k=3e5, classes=1000, data_dir="./data"):
    """
    creates shuffled, balanced subset of /benchmark_imglist/imagenet/train_imagenet.txt
    """

    save_path = data_dir + "/benchmark_imglist/imagenet/train_imagenet_subsample.txt"
    old_path = data_dir + "/benchmark_imglist/imagenet/train_imagenet.txt"
    read_path = data_dir + "/benchmark_imglist/imagenet/train_imagenet_shuffled.txt"

    if k == "full":
        os.system(f"cp {old_path} {save_path}")
        return

    # Shuffle the og file
    with open(old_path, "r") as old:
        lines = old.readlines()
        random.shuffle(lines)
        with open(read_path, "w") as new:
            new.writelines(lines)
        new.close()
    old.close()

    label_count = [0] * classes
    open(save_path, "w").close()

    with open(save_path, "w") as w:
        with open(read_path, "r") as r:
            while True:
                line = r.readline()
                if not line:
                    r.close()
                    w.close()
                    return

                splits = line.split()
                label = splits[1]

                if label_count[int(label)] > int(k / classes):
                    continue
                else:
                    label_count[int(label)] += 1

                w.write(line)

                if sum(label_count) >= k:
                    r.close()
                    w.close()
                    return


if __name__ == "__main__":
    start = time.time()
    generate_imglist()
    end = time.time()
    print(end - start)
