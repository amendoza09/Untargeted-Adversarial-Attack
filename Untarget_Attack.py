import numpy as np
import pickle
import matplotlib.pyplot as plt

def fgsm(x_test, y_test, model, eps=0.05):
    alpha=0.1 
    max_iterations=784
    learning_rate = 0.1

    # Generate a random image x0
    x0 = x_test

    # gradient descent
    for iteration in range(max_iterations):
        # Forward pass for model predictions
        logits = model.forward(x0)
        predicted_class = np.argmax(logits)

        # Calculate gradient of loss
        grad_x0 = model.grad_wrt_input(x0.reshape(1, -1), np.array([y_test]))

        # Update x0 with the gradient descent
        x0 = x0 + learning_rate * np.sign(grad_x0)

        if predicted_class != y_test:
            # Save image with filename
            filename = f"FGSM_untargeted.png"
            plt.imshow(x0.reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.savefig(filename)
            print(f"Generated image saved as '{filename}'")
            print("image classified as digit: {}".format(predicted_class))
            break
    return x0


def main():
    
    # load datasets
    mnist = None
    with open('mnist.pkl', 'rb') as fid:
        mnist = pickle.load(fid)
    # load model
    model = None
    with open('trained_model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    #visualize x_adv
    x_test = mnist['test_images'][0]
    y_test = mnist['test_labels'][0]
    fgsm(x_test, y_test, model)
    
if __name__ == "__main__":
    main()
