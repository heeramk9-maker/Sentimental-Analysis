import matplotlib.pyplot as plt

def plot_accuracy(acc_dict):
    models = list(acc_dict.keys())
    acc = list(acc_dict.values())

    plt.figure(figsize=(6,4))
    plt.bar(models, acc, color=["blue", "orange", "green"])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.show()
