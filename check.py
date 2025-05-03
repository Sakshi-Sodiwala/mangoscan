import os

# Define dataset directories
train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# Check if directories exist
print("Train directory exists:", os.path.exists(train_dir))
print("Val directory exists:", os.path.exists(val_dir))
print("Test directory exists:", os.path.exists(test_dir))

# Function to count images
def count_images(directory):
    count = 0
    for root, _, files in os.walk(directory):
        count += len([file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count

# Count images
print("Train images:", count_images(train_dir))
print("Val images:", count_images(val_dir))
print("Test images:", count_images(test_dir))
