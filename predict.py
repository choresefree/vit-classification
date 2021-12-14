import torch
from PIL import Image
import torch.utils.data
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch16_224


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_dict = {
        0: 'Black_footed_Albatross',
        1: 'Laysan_Albatross',
        2: 'Sooty_Albatross',
        3: 'Groove_billed_Ani',
        4: 'Crested_Auklet'
    }

    pre_tfs = transforms.Compose([transforms.Resize(250, Image.BILINEAR),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_path = 'test_imgs/4.png'
    image = Image.open(image_path).convert('RGB')
    image = pre_tfs(image).unsqueeze(0)

    # define and load model
    net = vit_small_patch16_224(pretrained=True)
    weight_path = 'checkpoint.pth'
    net.load_state_dict(torch.load(weight_path, map_location='cpu'))
    net = net.to(device)

    # predict image
    with torch.no_grad():
        output = net(image)
    prediction = output.argmax(dim=1).squeeze(0)
    print('prediction: ', class_dict[prediction.item()])


def main():
    test()


if __name__ == '__main__':
    main()
