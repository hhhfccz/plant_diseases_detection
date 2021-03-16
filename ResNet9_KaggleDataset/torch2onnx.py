import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_trans():
    model = torch.load("/home/liyucong/project/FarmGuard/ResNet_KaggleDataset/ResNet9_KaggleDataset.pth", map_location="cpu").to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, "ResNet9_KaggleDataset.onnx", export_params=True, verbose=True, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
    model_trans()

