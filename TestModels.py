import matplotlib.pyplot as plt

# Setup the scoring and optimization engines
criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
tbt_optimizer = optim.Adam(tbt_model.parameters(), lr=0.001)

epochs = 15

# Arrays to store the accuracy history for our graph
cnn_history = []
tbt_history = []

print("\n--- INITIATING TRAINING SEQUENCE & REAL-TIME TRACKING ---")
for epoch in range(epochs):
    cnn_model.train()
    tbt_model.train()

    # --- THE TRAINING LOOP ---
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Train CNN
        cnn_optimizer.zero_grad()
        cnn_outputs = cnn_model(images)
        cnn_loss = criterion(cnn_outputs, labels)
        cnn_loss.backward()
        cnn_optimizer.step()

        # Train Spatially-Bound TBT
        tbt_optimizer.zero_grad()
        tbt_glimpses = extract_zoned_glimpses(images, patch_size=7)
        tbt_outputs = tbt_model(tbt_glimpses)
        tbt_loss = criterion(tbt_outputs, labels)
        tbt_loss.backward()
        tbt_optimizer.step()

    # --- THE EVALUATION LOOP (Runs every epoch) ---
    cnn_correct, tbt_correct, total = 0, 0, 0
    cnn_model.eval()
    tbt_model.eval()

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            total += test_labels.size(0)

            # CNN Accuracy
            cnn_preds = torch.argmax(cnn_model(test_images), dim=1)
            cnn_correct += (cnn_preds == test_labels).sum().item()

            # TBT Accuracy
            tbt_test_glimpses = extract_zoned_glimpses(test_images, patch_size=7)
            tbt_preds = torch.argmax(tbt_model(tbt_test_glimpses), dim=1)
            tbt_correct += (tbt_preds == test_labels).sum().item()

    # Calculate and store the percentages
    cnn_epoch_acc = 100 * cnn_correct / total
    tbt_epoch_acc = 100 * tbt_correct / total
    cnn_history.append(cnn_epoch_acc)
    tbt_history.append(tbt_epoch_acc)

    print(f"Epoch {epoch+1} | CNN: {cnn_epoch_acc:.2f}% | TBT: {tbt_epoch_acc:.2f}%")

# --- GENERATE THE PUBLICATION GRAPH ---
print("\n--- GENERATING LEARNING CURVE ---")
plt.figure(figsize=(10, 6))

# Plot the lines
plt.plot(range(1, epochs + 1), cnn_history, label='CNN Monolith', color='#2ca02c', linewidth=2.5, marker='o')
plt.plot(range(1, epochs + 1), tbt_history, label='TBT Multi-Agent (Spatially Bound)', color='#ff7f0e', linewidth=2.5, marker='s')

# Format the graph to look like a professional research paper
plt.title('Learning Curve: Hierarchical vs. Multi-Agent Consensus Architecture', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Test Set Accuracy (%)', fontsize=12, fontweight='bold')
plt.xticks(range(1, epochs + 1))
plt.ylim(0, 105)
plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the graph
plt.show()
