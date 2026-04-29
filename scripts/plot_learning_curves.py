import os
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

epochs = [1, 2, 3, 4, 5]

# Image-only EfficientNet-B0
img_train_loss = [0.7736, 0.4814, 0.3476, 0.2927, 0.2413]
img_val_loss   = [0.7464, 0.7355, 0.8316, 0.7997, 1.0280]
img_train_acc  = [0.7736, 0.8995, 0.9250, 0.9362, 0.9480]
img_val_acc    = [0.8829, 0.9285, 0.9252, 0.9302, 0.9449]

# Multimodal EfficientNet-B0 + Metadata
mm_train_loss = [0.7940, 0.4773, 0.3837, 0.3175, 0.2629]
mm_val_loss   = [0.7918, 0.7034, 0.8491, 0.8712, 1.2587]
mm_train_acc  = [0.8228, 0.9031, 0.9122, 0.9195, 0.9419]
mm_val_acc    = [0.8829, 0.9067, 0.9178, 0.8940, 0.9569]

# Plot 1: Image-only loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, img_train_loss, marker="o", label="Train Loss")
plt.plot(epochs, img_val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Image-only EfficientNet-B0 Learning Curve (Loss)")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "image_only_learning_curve_loss.png"), dpi=200)
plt.close()

# Plot 2: Image-only accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, img_train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, img_val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Image-only EfficientNet-B0 Learning Curve (Accuracy)")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "image_only_learning_curve_accuracy.png"), dpi=200)
plt.close()

# Plot 3: Multimodal loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, mm_train_loss, marker="o", label="Train Loss")
plt.plot(epochs, mm_val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multimodal Learning Curve (Loss)")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "multimodal_learning_curve_loss.png"), dpi=200)
plt.close()

# Plot 4: Multimodal accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, mm_train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, mm_val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Multimodal Learning Curve (Accuracy)")
plt.xticks(epochs)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "multimodal_learning_curve_accuracy.png"), dpi=200)
plt.close()

print("Saved:")
print("outputs/figures/image_only_learning_curve_loss.png")
print("outputs/figures/image_only_learning_curve_accuracy.png")
print("outputs/figures/multimodal_learning_curve_loss.png")
print("outputs/figures/multimodal_learning_curve_accuracy.png")
