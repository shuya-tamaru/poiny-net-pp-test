import torch


def main():
    print("Hello, PointNet World!")
    print(f"PyTorch version: {torch.__version__}")

    # GPUが使用可能か確認
    print(f"CUDA available: {torch.cuda.is_available()}")

    # テンソル作成のデモ
    x = torch.rand(5, 3)
    print("Random Tensor:")
    print(x)


if __name__ == "__main__":
    main()
