import torch
import torch.nn as nn
from torch.optim import AdamW
from nemo_fusion.multimodal import (
    MultiModalFusion,
    MultiModalDataset,
    MultiModalDataLoader,
    ModalityConfig,
)


class SimpleMultiModalModel(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 1024,
        hidden_dim: int = 768,
        num_classes: int = 10,
        fusion_type: str = "cross_attention",
    ):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fusion = MultiModalFusion(
            modality_dims={"text": hidden_dim, "image": hidden_dim},
            output_dim=hidden_dim,
            fusion_type=fusion_type,
            num_heads=8,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch):
        text_features = self.text_encoder(batch["text"])
        image_features = self.image_encoder(batch["image"])
        fused_features = self.fusion({"text": text_features, "image": image_features})
        pooled_features = fused_features.mean(dim=1)
        return self.classifier(pooled_features)


def create_dummy_dataset(num_samples: int = 1000):
    return [{
        "text": torch.randn(128, 768),
        "image": torch.randn(196, 1024),
        "label": i % 10,
    } for i in range(num_samples)]


def main():
    print("Multi-Modal Training with NeMo Fusion\n")

    batch_size, num_epochs, lr = 8, 3, 1e-4

    train_data = create_dummy_dataset(800)
    val_data = create_dummy_dataset(200)

    modality_configs = {
        "text": ModalityConfig(name="text", data_type="text"),
        "image": ModalityConfig(name="image", data_type="image"),
    }

    train_dataset = MultiModalDataset(data=train_data, modality_configs=modality_configs)
    val_dataset = MultiModalDataset(data=val_data, modality_configs=modality_configs)

    train_loader = MultiModalDataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = MultiModalDataLoader(val_dataset, batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = SimpleMultiModalModel(768, 1024, 768, 10, "cross_attention")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = nn.functional.cross_entropy(outputs, batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {i}, Loss: {loss.item():.4f}")

    print("\nInference example:")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
        outputs = model(sample_batch)
        predictions = outputs.argmax(dim=-1)
        print(f"Predictions: {predictions[:5].tolist()}")
        print(f"Labels: {sample_batch['label'][:5].tolist()}")


if __name__ == "__main__":
    main()

