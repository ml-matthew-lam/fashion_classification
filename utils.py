import torch

def calculate_accuracy(predictions, labels):
    """
    Calculates % accuracy of predictions
    """
    top_guesses = torch.max(predictions, dim=1)[1] # get index of max score (prediction)
    correct = (top_guesses == labels).sum().item() # sum the number of correct predictions
    return (correct/labels.size(0))*100 #return the percentage

def save_model(model, filepath = "fashion_model.pth"):
    """
    Saves learned weights in a file
    """
    torch.save(model.state_dict(), filepath)