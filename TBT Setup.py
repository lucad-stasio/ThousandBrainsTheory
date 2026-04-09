def extract_zoned_glimpses(images, patch_size=7):
    """
    Divides the 28x28 image into a perfect 4x4 grid of 16 sectors.
    Extracts exactly one 7x7 patch per sector.
    Returns an array of shape: [batch_size, 16, 51] (49 pixels + X + Y)
    """
    batch_size = images.shape[0]
    num_patches = 28 // patch_size # 4 patches per row/col
    total_glimpses = num_patches * num_patches # 16 glimpses

    # Initialize the empty tensor to hold the glimpses
    glimpses = torch.zeros((batch_size, total_glimpses, patch_size * patch_size + 2)).to(device)

    idx = 0
    for row in range(num_patches):
        for col in range(num_patches):
            # Calculate the exact locked coordinates for this sector
            y = row * patch_size
            x = col * patch_size

            # Slice the 7x7 patch from this specific sector
            patch = images[:, 0, y:y+patch_size, x:x+patch_size]
            patch_flat = patch.reshape(batch_size, -1)

            # Normalize X and Y coordinates to tell the agent where it is looking
            x_norm = torch.full((batch_size, 1), x / 28.0).to(device)
            y_norm = torch.full((batch_size, 1), y / 28.0).to(device)

            # Append X and Y to the pixel data
            glimpse_tensor = torch.cat((patch_flat, x_norm, y_norm), dim=1)
            glimpses[:, idx, :] = glimpse_tensor
            idx += 1

    return glimpses
