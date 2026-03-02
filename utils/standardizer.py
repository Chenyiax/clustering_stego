import torch

class TensorStandardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def standardize(self, tensor):
        # 在标准化时计算均值和标准差
        self.mean = tensor.mean()
        self.std = tensor.std()
        standardized_tensor = (tensor - self.mean) / self.std
        return standardized_tensor

    def restore(self, tensor):
        if self.mean is None:
            raise ValueError("Tensor has not been standardized yet.")
        restored_tensor = tensor * self.std + self.mean
        return restored_tensor

if __name__ == "__main__":
    # 使用示例
    tensor = torch.randn(10)
    standardizer = TensorStandardizer()
    print("原始张量:", tensor)
    # 标准化
    standardized_tensor = standardizer.standardize(tensor)
    print("标准化后的张量:", standardized_tensor)

    # 还原
    restored_tensor = standardizer.restore(standardized_tensor)
    print("还原后的张量:", restored_tensor)
    print("是否还原成功:", torch.allclose(tensor, restored_tensor))
