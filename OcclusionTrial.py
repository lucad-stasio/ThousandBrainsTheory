# --- THE OCCLUSION STRESS TEST ---
print("\n--- RUNNING OCCLUSION (DAMAGE) PROTOCOL ---")

def apply_occlusion(images, block_size=10):
    """
    Simulates a blocked sensor or obscured object by
    drawing a solid black square over a random part of the image.
    """
    corrupted_images = images.clone()
    batch_size = images.shape[0]

    for b in range(batch_size):
        # Pick a random starting point for the blackout square
        x = random.randint(0, 28 - block_size)
        y = random.randint(0, 28 - block_size)

        # Set all pixels in that square to 0 (Black)
        corrupted_images[b, 0, y:y+block_size, x:x+block_size] = 0

    return corrupted_images

cnn_occ_correct, tbt_occ_correct, total = 0, 0, 0

cnn_model.eval()
tbt_model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)

        # Corrupt the images before testing
        damaged_images = apply_occlusion(images, block_size=12) # Blocks out almost half the image

        # --- CNN Damage Accuracy ---
        cnn_preds = torch.argmax(cnn_model(damaged_images), dim=1)
        cnn_occ_correct += (cnn_preds == labels).sum().item()

        # --- TBT Damage Accuracy ---
        # The Slicer extracts glimpses from the DAMAGED image
        tbt_glimpses = extract_zoned_glimpses(damaged_images, patch_size=7)
        tbt_preds = torch.argmax(tbt_model(tbt_glimpses), dim=1)
        tbt_occ_correct += (tbt_preds == labels).sum().item()

print(f"CNN Accuracy under severe occlusion: {100 * cnn_occ_correct / total:.2f}%")
print(f"TBT Accuracy under severe occlusion: {100 * tbt_occ_correct / total:.2f}%")
