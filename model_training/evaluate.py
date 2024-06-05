from matplotlib import pyplot as plt
import torch
import sklearn
import seaborn as sns


# Test the model and report the classification metrics
def report_metrics(loader, model, device, title, class_labels):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_true.extend(labels.tolist())

            outputs = torch.softmax(model(inputs), dim=1)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

    class_report = sklearn.metrics.classification_report(y_true, y_pred)
    print(f"Classification report for the {title}:")
    print(class_report)

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for the {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
