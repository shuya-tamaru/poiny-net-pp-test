import torch
import sys
from src.core.learning import learning
from src.core.segment import segment

NUM_CLASSES = 12


def check_if_cuda_available():
    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available.")


def main(mode="learning"):
    check_if_cuda_available()
    if mode == "learning":
        learning(num_classes=NUM_CLASSES)
    else:
        segment(num_classes=NUM_CLASSES)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    main(mode)
