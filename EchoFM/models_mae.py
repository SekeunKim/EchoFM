# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from EchoFM.util import video_vit
from EchoFM.util.logging import master_print as print
import torch.nn.functional as F

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_prj_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        
        self.decoder_block = video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
        
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.initialize_weights()
        
        
        print("model initialized")

    def self_similarity(self, cls_tokens):
        """
        Compute self-similarity map using cosine similarity.
        
        Args:
            cls_tokens (list of tensors): List of tensors, where each tensor is of shape [N, D].
        
        Returns:
            similarity_map (tensor): Tensor of shape [N, T, T] containing self-similarity values.
        """
        # Concatenate the list into a single tensor of shape [N, T, D]
        cls_tokens_tensor = torch.stack(cls_tokens, dim=1)  # Shape: [N, T, D]

        # Normalize embeddings to unit vectors
        cls_tokens_tensor = F.normalize(cls_tokens_tensor, p=2, dim=-1)  # Shape: [N, T, D]
        
        # Compute cosine similarity
        similarity_map = torch.matmul(cls_tokens_tensor, cls_tokens_tensor.transpose(1, 2))  # Shape: [N, T, T]
        
        return similarity_map

    def triplet_sampling(self, similarity_map, cls_tokens):
        """
        Perform triplet sampling with one anchor, one positive, and one negative per batch.

        Args:
            similarity_map (tensor): Self-similarity map of shape [N, T, T].
            cls_tokens (tensor): Tensor of CLS tokens, shape [N, T, D].

        Returns:
            anchor (tensor): Tensor of anchor embeddings, shape [N, D].
            positive (tensor): Tensor of positive embeddings, shape [N, D].
            negative (tensor): Tensor of negative embeddings, shape [N, D].
        """
        
        cls_tokens = torch.stack(cls_tokens, dim=1)  # Shape: [N, T, D]
         
        N, T, D = cls_tokens.shape

        anchors, positives, negatives = [], [], []

        for n in range(N):  # Iterate over batches
            # Extract the first row (anchor is always index 0)
            first_row = similarity_map[n, 0, :]  # Shape: [T]

            # Compute mean similarity for the first row
            mean_similarity = first_row.mean().item()

            # Identify positive and negative indices, excluding anchor index (0)
            positive_indices = (first_row > mean_similarity).nonzero(as_tuple=True)[0]
            positive_indices = positive_indices[positive_indices != 0]  # Exclude anchor (index 0)

            negative_indices = (first_row <= mean_similarity).nonzero(as_tuple=True)[0]
            negative_indices = negative_indices[negative_indices != 0]  # Exclude anchor (index 0)

            # Ensure we have at least one positive and one negative
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Randomly select one positive and one negative
                pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))].item()
                neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))].item()

                # Append CLS tokens for the selected indices
                anchors.append(cls_tokens[n, 0, :])  # Anchor is always index 0
                positives.append(cls_tokens[n, pos_idx, :])  # Positive CLS token
                negatives.append(cls_tokens[n, neg_idx, :])  # Negative CLS token

        # Stack tensors to create final batch outputs
        anchor = torch.stack(anchors)  # Shape: [N, D]
        positive = torch.stack(positives)  # Shape: [N, D]
        negative = torch.stack(negatives)  # Shape: [N, D]

        return anchor, positive, negative
    
    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def uniform_random_masking(self, x, mask_ratio, L):
        """
        Perform temporal consistent random masking by sampling the same spatial tokens across time steps.
        Args:
            x: Tensor of shape [N, T * L, D], sequence after patch embedding (flattened temporal and spatial dimensions).
            mask_ratio: Float, proportion of tokens to mask.
            L: Number of spatial tokens per time step.

        Returns:
            x_masked: Tensor of shape [N, len_keep * T, D], after masking.
            mask: Binary mask of shape [N, T * L], 0 is keep, 1 is remove.
            ids_restore: Indices to restore original sequence order.
            ids_keep: Indices of kept tokens.
        """
        N, TL, D = x.shape  # Batch size, total tokens, embedding dimension
        T = TL // L  # Temporal length

        # Compute the number of tokens to keep per spatial location
        len_keep = int(L * (1 - mask_ratio))

        # Generate random noise for each spatial location
        noise = torch.rand(N, L, device=x.device)  # [N, L]

        # Sort spatial tokens based on noise
        ids_shuffle = torch.argsort(noise, dim=1)  # [N, L]
        ids_keep = ids_shuffle[:, :len_keep]  # Keep top len_keep indices [N, len_keep]
        ids_keep = ids_keep.unsqueeze(1).repeat(1, T, 1)  # Broadcast to all time steps [N, T, len_keep]

        # Create a binary mask for all time steps
        mask = torch.ones(N, T, L, device=x.device)  # Initialize mask with all 1s [N, T, L]
        
        for n in range(N):  # Iterate over batch
            for t in range(T):
                mask[n, t, ids_keep[n]] = 0  # Use batch-specific ids_keep[n]

        mask = mask.view(N, TL)  # Flatten to [N, T * L]
        ids_restore = torch.argsort(mask, dim=1)  # Indices for restoring order

        # Mask input
        x_masked = x[mask == 0].view(N, -1, D)  # Kept tokens only [N, len_keep * T, D]

        ids_keep = ids_keep.view(N, -1)
        return x_masked, mask, ids_restore, ids_keep

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        
        x, mask, ids_restore, ids_keep = self.uniform_random_masking(x, mask_ratio, L)
        
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore
    
    def decoder_prj(self, x):
        # apply Transformer blocks
        x = self.decoder_block(x)
        x = self.norm(x)

        if self.cls_embed:
            return x[:, 0, :]
        else:
            print ('CLS token is needed')
        
    
    def forward_prj(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size
        
        # embed tokens (divide to temporal)
        
        # x = 4 392 1024 
        
        # x = reshape() -> 4 8 49 1024
        x = x.view(N, T, 49, 1024) 
        
        cls_ = []
        for i in range(T):
            x_t = x[:,i,:,:]
            
            if self.cls_embed:
                decoder_cls_token = self.decoder_prj_cls_token
                decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
                x_t = torch.cat((decoder_cls_tokens, x_t), dim=1)
                 
            x_t_cls = self.decoder_prj(x_t) #vit
            cls_.append(x_t_cls)
        return cls_
    
    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss 

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        cls_tokens = self.forward_prj(latent, ids_restore)
        
        similarity_map = self.self_similarity(cls_tokens)
        
        anchor, positive, negative = self.triplet_sampling(similarity_map, cls_tokens)
        
        # triplet sampling
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        
        loss = loss + triplet_loss
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
