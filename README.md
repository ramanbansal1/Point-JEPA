# Point-JEPA : Joint-Embedding Predictive Architecture for Point Cloud Completion

Point-JEPA is an experimental framework for self-supervised 3D point cloud understanding and completion, inspired by JEPA-style predictive learning.
The core idea is to predict missing geometric regions in latent space, then decode them into dense point clouds using a lightweight, geometry-aware decoder.

This repository focuses on encoder quality evaluation, not heavy generative modeling.

## Overview

Point-JEPA consists of two main components:

- Dual JEPA Encoder
- Encodes visible context points
- Predicts latent tokens for masked / missing regions
- Trained using JEPA-style predictive objectives (no reconstruction loss)
- Point Decoder (Inference / Evaluation only)
- Converts context tokens and predicted tokens into a dense point cloud
- Uses non-trainable geometric upsampling
- Uses shared EdgeConv layers for local geometric reasoning

Produces geometry via offset prediction (not absolute coordinates)

The decoder is intentionally simple, stable, and interpretable, making it ideal for testing whether the encoder has learned meaningful geometry.

## Architecture
```
High-level pipeline
Input Point Cloud
        │
        ▼
Context Sampling (P points)
        │
        ▼
JEPA Encoder
 ├── Context Tokens
 └── Predicted Tokens (for masked regions)
        │
        ▼
Point Decoder
 ├── Context Upsampling (non-trainable)
 ├── Pred Token Seeding
 ├── Shared EdgeConv Geometry Reasoning
 ├── Offset-based Coordinate Updates
        │
        ▼
Dense Point Cloud
```
