class TBT_Column(nn.Module):
    def __init__(self):
        super(TBT_Column, self).__init__()

        # --- THE BASAL DENDRITES (Spatial Processing) ---
        # Takes the 2 (X,Y) coordinates and expands them into a 49-number spatial mask
        self.location_layer = nn.Linear(2, 49)

        # --- THE CELL BODY ---
        # The input is now exactly 49, because the 49 pixels will be gated
        # by the 49 spatial numbers BEFORE hitting this layer.
        self.fc1 = nn.Linear(49, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, glimpse):
        # 1. Separate the WHAT (pixels) and the WHERE (coordinates)
        pixels = glimpse[:, :49]
        coords = glimpse[:, 49:]

        # 2. Process the WHERE into a spatial context mask
        # We use a Sigmoid function so the spatial weights are between 0 and 1
        spatial_context = torch.sigmoid(self.location_layer(coords))

        # 3. COINCIDENCE DETECTION (The NMDA Spike)
        # We mathematically multiply the sensory pixels by the spatial context.
        # The WHERE physically alters the WHAT before the brain processes it.
        bound_input = pixels * spatial_context

        # 4. Process the bound signal
        x = self.relu(self.fc1(bound_input))
        x = self.fc2(x)
        return x

class TBT_Network(nn.Module):
    def __init__(self, num_columns=16):
        super(TBT_Network, self).__init__()
        self.num_columns = num_columns
        self.columns = nn.ModuleList([TBT_Column() for _ in range(num_columns)])

    def forward(self, glimpses):
        batch_size = glimpses.shape[0]
        num_glimpses = glimpses.shape[1]

        weighted_votes = torch.zeros((batch_size, 10)).to(device)
        total_confidence = torch.zeros((batch_size, 1)).to(device)

        for column in self.columns:
            for j in range(num_glimpses):
                glimpse = glimpses[:, j, :]
                raw_vote = column(glimpse) # Glimpse splitting happens inside the column now

                # Confidence Weighting (Lateral Inhibition)
                probabilities = torch.softmax(raw_vote, dim=1)
                confidence_score, _ = torch.max(probabilities, dim=1, keepdim=True)
                weighted_votes += raw_vote * confidence_score
                total_confidence += confidence_score

        return weighted_votes / total_confidence

tbt_model = TBT_Network(num_columns=16).to(device)
print(f"Spatially-Bound TBT Parameters: {sum(p.numel() for p in tbt_model.parameters())}")
