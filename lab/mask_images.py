from PIL import Image
import os

# Set your directories
rgb_dir = '../SAM6D_FULL/SAM-6D/SAM-6D/Data/BOP-Templates/tless/obj_000011'
mask_dir = '../SAM6D_FULL/SAM-6D/SAM-6D/Data/BOP-Templates/tless/obj_000011'
output_dir = './masked_obj'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image
for i in range(42):
    # Construct the file paths
    rgb_path = os.path.join(rgb_dir, f'rgb_{i}.png')
    mask_path = os.path.join(mask_dir, f'mask_{i}.png')
    output_path = os.path.join(output_dir, f'marked_{i}.png')

    # Open the RGB and mask images
    rgb_image = Image.open(rgb_path).convert("RGBA")
    mask_image = Image.open(mask_path).convert("L")  # Convert mask to grayscale (L mode)

    # Create an alpha mask by applying the grayscale mask to each pixel
    # Make pixels transparent (alpha=0) where mask is black (0)
    rgba_image = rgb_image.copy()
    rgba_image.putalpha(mask_image)

    # Save the output image with transparency
    rgba_image.save(output_path, format="PNG")

print("Processing complete. Marked images are saved as transparent PNGs.")
