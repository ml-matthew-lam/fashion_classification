import torch
import torchvision.transforms as transforms
from PIL import Image
from model import FashionModel
import matplotlib.pyplot as plt


# filename here
IMAGE_PATH = "ankle_boot_dark.png"


def preprocess_image(image_path):
    """reshapes an image to be inputted to the network"""
    img = Image.open(image_path)

    transformations = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1), # make image grayscale
        transforms.Resize((28,28)), # resize to 28x28
        transforms.ToTensor(), # convert to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,)) # shift and stretch pixel scale to between -1 and +1
    ])
    return transformations(img).unsqueeze(0) # this also adds a "batch" dimension since the model takes in batches

if __name__ == "__main__":
    model = FashionModel()
    model.load_state_dict(torch.load('fashion_model_weights.pth', map_location = torch.device('cpu'), weights_only = True))
    model.eval()

    classes = {0: 't-shirt/top', 1: 'trouser', 2: 'pullover', 3: 'dress', 4: 'coat', 5: 'sandal', 6: 'shirt', 7: 'sneaker', 8: 'bag', 9: 'ankle boot'}

    tensor_image = preprocess_image(IMAGE_PATH)
    with torch.no_grad():
        logit_output = model(tensor_image) # logit_output is of shape [1, 10] (= 1 image in the batch, 10 logits per image)
        prob_output = torch.softmax(logit_output, dim = 1)[0]
    
    max_idx = torch.argmax(prob_output).item()


    # all the following lines are to display the pre-processed image along with the predictions of the model

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    display_img = tensor_image.squeeze() / 2 + 0.5 # Un-normalize back to 0.0 - 1.0 scale
    ax1.imshow(display_img, cmap='gray')
    ax1.axis('off') # Remove background chart coordinates

    class_names = [classes[i] for i in range(10)]
    bars = ax2.barh(class_names, prob_output * 100, color='lightgray')
    
    bars[max_idx].set_color('#1f77b4')
    
    ax2.set_xlabel('probability', fontweight='bold')
    ax2.set_xlim(0, 100) # Ensure full percentage bounds
    ax2.grid(axis='x', linestyle='--', alpha=0.5) # Add soft grid backdrop

    plt.tight_layout()
    plt.show()




    

