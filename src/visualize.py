import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, filename="visualizations/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Confusion matrix saved to: {filename}")
    plt.close()

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print("\n📋 Classification Report:")
    print(report)
    return report
