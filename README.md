# Adaptive-PADRe

```
Input: x (Tensor) -> Shape: [B, 9, 128]
│
├── 1. input_proj (Stem / Projection Layer)
│   ├── nn.Conv1d(in_channels=9, out_channels=48, kernel_size=1)
│   └── Output Shape: [B, 48, 128]
│
├── 2. padre_blocks (Encoder / nn.ModuleList, 3 Layers 반복)
│   │
│   ├── [Layer i: PADReBlock] Input Shape: [B, 48, 128]
│   │   │
│   │   ├── res = x (Residual 저장)
│   │   │
│   │   ├── degree_gate: ComputeAwareDegreeGate
│   │   │   ├── nn.AdaptiveAvgPool1d(1)
│   │   │   ├── nn.Linear(48, 16) -> nn.GELU() -> nn.Linear(16, 3)
│   │   │   ├── Softmax & STE (Straight-Through Estimator)
│   │   │   └── Output: dw, logits, sp / sel = dw.argmax(dim=-1)
│   │   │
│   │   ├── _hard_forward & _build_Z (Polynomial Hadamard Mixing)
│   │   │   ├── channel_mixing: nn.ModuleList([nn.Conv1d(48, 48, k=1)])
│   │   │   ├── token_mixing: nn.ModuleList([nn.Conv1d(48, 48, k=11, groups=48)])
│   │   │   ├── pre_hadamard_channel: nn.ModuleList([nn.Conv1d(48, 48, k=1)])
│   │   │   ├── pre_hadamard_token: nn.ModuleList([nn.Conv1d(48, 48, k=11, groups=48)])
│   │   │   ├── Z_stack 생성 및 고급 인덱싱: Z_stack[sel, torch.arange(B)]
│   │   │   └── Output Shape: [B, 48, 128]
│   │   │
│   │   ├── norm: nn.LayerNorm(48)
│   │   │
│   │   ├── Residual Add 1 & LayerNorm
│   │   │   └── x = _ln(norms1[i], x + res)
│   │   │
│   │   ├── res2 = x (Residual 저장)
│   │   │
│   │   ├── ffn[i]: nn.Sequential (Feed-Forward Network)
│   │   │   ├── nn.Conv1d(48, 96, kernel_size=1)
│   │   │   ├── nn.GELU()
│   │   │   ├── nn.Dropout(p=0.2)
│   │   │   ├── nn.Conv1d(96, 48, kernel_size=1)
│   │   │   └── nn.Dropout(p=0.2)
│   │   │
│   │   └── Residual Add 2 & LayerNorm
│   │       └── x = _ln(norms2[i], x + res2)
│   │
│   └── Output Shape (after 3 layers): [B, 48, 128]
│
├── 3. global_pool & flatten
│   ├── nn.AdaptiveAvgPool1d(1)  -> Shape: [B, 48, 1]
│   ├── .squeeze(-1)             -> Shape: [B, 48]
│   └── Output Shape: [B, 48]
│
└── 4. classifier (Classification Head / MLP)
    ├── nn.Sequential
    │   ├── nn.Linear(in_features=48, out_features=48)
    │   ├── nn.LayerNorm(48)
    │   ├── nn.GELU()
    │   ├── nn.Dropout(p=0.2)
    │   └── nn.Linear(in_features=48, out_features=6)
    │
    └── Final Output (logits): [B, 6]
```
