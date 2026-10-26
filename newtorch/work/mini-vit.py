from torchvision import datasets
import matplotlib.pyplot as plt

train_set = datasets.CIFAR10(root="./data", train=True, download=False)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 每类挑一张，凑 10 张不同类别的图
seen = {}
for i, (img, label) in enumerate(train_set):
    if label not in seen:
        seen[label] = img
    if len(seen) == 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, label in zip(axes.flat, range(10)):
    # NEAREST 插值放大，能看到真实像素；其他插值会虚假地"变平滑"
    ax.imshow(seen[label].resize((160, 160), resample=0))
    ax.set_title(class_names[label], fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150)
plt.show()
print("saved to cifar10_samples.png")
