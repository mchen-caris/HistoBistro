from pathlib import Path
import pytorch_lightning as pl
import yaml
import argparse
import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import matplotlib.patches as patches
from tqdm import tqdm

from matplotlib.colors import LinearSegmentedColormap

from classifier import ClassifierLightning
from options import Options


# %% md
### Load plotting utils
# %%

# function that plots scores nicely. scores should have the same length as the number of tiles.
def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())


def plot_scores(coords, scores, image, overlay=True, clamp=0.05, norm=True, colormap='RdBu', crop=False, indices=[]):
    if clamp:
        q05, q95 = torch.quantile(scores, clamp), torch.quantile(scores, 1 - clamp)
        scores.clamp_(q05, q95)

    if norm:
        scores = NormalizeData(scores)

    if crop:
        coords_min, coords_max = np.array(coords).min(axis=0), np.array(coords).max(axis=0)
        y_min, y_max, x_min, x_max = round(coords_min[1] / d), round(coords_max[1] / d), round(
            coords_min[0] / d), round(coords_max[0] / d)
        if slide_path.stem == '439042':
            x_max = round((69 * 1013) / d)
        print(y_min, y_max, x_min, x_max)
    else:
        y_min, y_max, x_min, x_max = 0, image.shape[0], 0, image.shape[1]

    attention_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    tissue_map = -np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

    offset = 1013
    for (x, y), s in zip(coords, scores):

        if colormap == 'RdBu':
            attention_map[round(y / d):round((y + offset) / d), round(x / d):round((x + offset) / d)] = 1 - s.item()
        else:
            attention_map[round(y / d):round((y + offset) / d), round(x / d):round((x + offset) / d)] = s.item()
        tissue_map[round(y / d):round((y + offset) / d), round(x / d):round((x + offset) / d)] = s.item()

    attention_map = np.array(attention_map * 255., dtype=np.uint8)
    tissue_map[tissue_map >= 0] = 1
    tissue_map[tissue_map < 0] = 0

    if len(indices) != 0:
        highlight_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        for i in indices:
            x, y = coords[i]
            highlight_map[round(y / d):round((y + offset) / d), round(x / d):round((x + offset) / d)] = 1

        #     plt.figure(figsize=(30, 30))
    a = 1.
    if overlay:
        plt.imshow(image[y_min:y_max, x_min:x_max])
        a = 0.5

    if crop:
        plt.imshow(attention_map[y_min:y_max, x_min:x_max], alpha=a * (tissue_map[y_min:y_max, x_min:x_max]),
                   cmap=colormap, interpolation='nearest')
    #         plt.imshow(attention_map[round(coords_min[1]/d):, round(coords_min[0]/d):], alpha=a*(tissue_map[round(coords_min[1]/d):, round(coords_min[0]/d):]), cmap=colormap, interpolation='nearest')
    else:
        plt.imshow(attention_map, alpha=a * (tissue_map), cmap=colormap, interpolation='nearest')

    if len(indices) != 0:
        plt.imshow(highlight_map[y_min:y_max, x_min:x_max], alpha=1. * (highlight_map), cmap='viridis',
                   interpolation='nearest')

    plt.axis('off')


# %% md
### Load attention utils
# %%
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


# %%
def generate_rollout(model, input, start_layer=0):
    model(input)
    blocks = model.transformer.layers
    all_layer_attentions = []
    for blk in blocks:
        attn_heads = blk[0].fn.get_attention_map()
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    return rollout[:, 0, 1:]


# %% md
### Load model weights
# %%
parser = Options()
args = parser.parser.parse_args('')

# --------------------------------------------
# TODO: add custom data
# args.config_file = '/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-10-30--12-49-36/yaml_files/aws_config_MSI--MSI.yaml'  # heads: 2
args.config_file = '/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--15-09-54--12epochs/yaml_files/aws_config_MSI--MSI.yaml'  # heads: 8
# --------------------------------------------

# Load the configuration from the YAML file
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# # Update the configuration with the values from the argument parser
# for arg_name, arg_value in vars(args).items():
#     if arg_value is not None and arg_name != 'config_file':
#         config[arg_name]['value'] = getattr(args, arg_name)

# Create a flat config file without descriptions
config = {k: v['value'] for k, v in config.items()}

print('\n--- load options ---')
for name, value in sorted(config.items()):
    print(f'{name}: {str(value)}')

cfg = argparse.Namespace(**config)
# %%
name = 'multi-all-cohorts-every1000'
target = 'isMSIH'
fold = 2
# # BRAF experiments
# name = 'multi-all-cohorts'
# target = 'BRAF'
# fold = 3
# # KRAS experiments
# name = 'multi-all-cohorts'
# target = 'KRAS'
# fold = 2
# model_path = Path(f'/Users/sophia.wagner/Documents/PhD/projects/2022_MSI_transformer/attention-user-study/multi-all-cohorts-every1000_transformer_DACHS-QUASAR-RAINBOW-TCGA_histaugan_isMSIH/models/best_model_multi-all-same_transformer_DACHS-QUASAR-RAINBOW-TCGA_histaugan_isMSIH_fold3.ckpt')
# model_path = Path(f'/Volumes/SSD/logs/idkidc/multi-all-cohorts-every1000_transformer_CPTAC-DACHS-DUSSEL-Epi700-ERLANGEN-FOXTROT-MCO-MECC-MUNICH-QUASAR-RAINBOW-TCGA-TRANSCOT_histaugan_isMSIH/models/best_model_{name}_transformer_CPTAC-DACHS-DUSSEL-Epi700-ERLANGEN-FOXTROT-MCO-MECC-MUNICH-QUASAR-RAINBOW-TCGA-TRANSCOT_histaugan_isMSIH_fold{fold}.ckpt/')
# model_path = Path(f'/Volumes/SSD/logs/idkidc/multi-all-cohorts_transformer_DACHS-QUASAR-RAINBOW-TCGA-MCO_histaugan_BRAF/models/best_model_{name}_transformer_DACHS-QUASAR-RAINBOW-TCGA-MCO_histaugan_BRAF_fold{fold}.ckpt/')
# model_path = Path(
#     f'/Volumes/SSD/logs/idkidc/multi-all-cohorts_transformer_DACHS-QUASAR-RAINBOW-TCGA-MCO_histaugan_{target}/models/best_model_{name}_transformer_DACHS-QUASAR-RAINBOW-TCGA-MCO_histaugan_{target}_fold{fold}.ckpt/')
model_path = Path(
    '/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--15-09-54--12epochs/output/logs/local-Transformer-CRC_all-MSI--MSI-lr1e-05-wd1e-05_Transformer_caris_raw_MSI--MSI/models/best_model_local-Transformer-CRC_all-MSI--MSI-lr1e-05-wd1e-05_Transformer_caris_raw_MSI--MSI_fold0.ckpt')
# cfg.pos_weight = torch.tensor([1.0])
cfg.pos_weight = torch.tensor([])
classifier = ClassifierLightning(cfg)
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
checkpoint['state_dict'].keys()
classifier.load_state_dict(checkpoint['state_dict'])
classifier.eval();
# %% md
### Load features and slides
# %%
# slide_csv = Path('/Users/sophia.wagner/Documents/PhD/data/YCR-BCIP/YORKSHIRE-RESECTIONS-DX_SLIDE.csv')
# slide_csv = Path('/Users/sophia.wagner/Documents/PhD/data/Epi700/BELFAST-CRC-DX_SLIDE.csv')
slide_csv = Path('/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/slide_table_MSI--MSI.csv')
slide_csv = pd.read_csv(slide_csv)
# %%
# patient_id = Path('18-LSS0736') 439097.h5
# base_dir = Path('/Users/sophia.wagner/Documents/PhD/data/YCR-BCIP/attention_study')
# slide_dir = base_dir / 'slides'
# base_dir = Path('/Users/sophia.wagner/Documents/PhD/data/YCR-BCIP')
# slide_dir = base_dir / 'attention_visualization'
base_dir = Path('/Users/sophia.wagner/Documents/PhD/data/Epi700')
slide_dir = base_dir
slides = list(slide_dir.glob('*.svs'))
slides.sort()

# %%
# slides.index(slide_dir / '439097.svs')
# %%
slide_idx = 1
slide_path = slides[slide_idx]
# feature_dir = base_dir  # / 'attention_visualization' #  / 'features'
feature_dir = Path('/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/fests_ctrans')
# feature_path = feature_dir / f'{slide_path.stem}.h5'
feature_path = feature_dir / f'00369d04-d0f4-4762-a3f6-c02997ede9d7.h5'
print(slide_path.name)
print(slides)
# %%
h5_file = h5py.File(feature_path)
features = torch.Tensor(np.array(h5_file['feats'])).unsqueeze(0)
coords = torch.Tensor(np.array(h5_file['coords']))
coords = [(coords[i, 0].int().item(), coords[i, 1].int().item()) for i in range(coords.shape[0])]
# %%
slide = openslide.OpenSlide(slide_path)
level = len(slide.level_downsamples) - 1
d = slide.level_downsamples[level]
image = slide.read_region((0, 0), level, slide.level_dimensions[level])
# %%
image = np.array(image.convert("RGB"))
# %%
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
# %% md
### Compute attention scores
# %%
rollout = generate_rollout(classifier.model, features, start_layer=0).squeeze(0)
# %%
plot_scores(coords, rollout, image, overlay=True, colormap='viridis', crop=True)
plt.show()
# %% md
### Compute class scores
# %%
n = features.shape[1]
scores = np.zeros(n)
for i in tqdm(range(n)):
    out = classifier.model(features[:, i:i + 1, :]).squeeze(0)
    scores[i] = torch.sigmoid(out)
# %%
plot_scores(coords, scores, image, overlay=True, colormap='RdBu_r', clamp=False, norm=False, crop=True)
plt.show()
# %% md
### attention x class scores
# %%
plot_scores(coords, rollout * scores, image, overlay=True, colormap='RdBu_r', clamp=False, norm=True)
plt.show()
# %% md
## User study on high attention tiles
# %% md
### prepare user study
# %%
k = 100
values, indices = rollout.topk(k)
l = 10
every_l = [indices[i] for i in range(k) if i % l == 0]
# %%
indices
# %%
every_l
# %%
plot_scores(coords, rollout, image, indices=indices)
plt.show()
# %%
n = features.shape[1]
scores = torch.zeros(n)
for i in tqdm(range(n)):
    out = classifier.model(features[:, i:i + 1, :]).squeeze(0)
    scores[i] = torch.sigmoid(out)
# %%
plot_scores(coords, scores, image, norm=False, clamp=0., colormap='viridis', overlay=False)
# %%
torch.tensor(scores).topk(10, largest=False)
# %%
scores.min(), scores.max()
# %%
rollout[indices]
# %%
for i in indices:
    print(scores[i])
# %%
# plot top tiles
ind = indices[:10]

rows, columns, img_size = 2, 5, 5
plt.figure(figsize=(columns * img_size, rows * (img_size + 0.5)))
for i in range(len(ind)):
    x, y = coords[ind[i]]
    margin = 250
    size = 1013
    tile = slide.read_region((x - margin, y - margin), 0, (size + 2 * margin, size + 2 * margin))

    plt.subplot(rows, columns, i + 1)
    plt.imshow(tile)
    plt.title(i)

    box_x = margin  # x-coordinate of the top-left corner of the rectangle
    box_y = margin  # y-coordinate of the top-left corner of the rectangle
    box_width = size  # Width of the rectangle
    box_height = size  # Height of the rectangle

    rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    plt.axis('off')
plt.tight_layout()
# plt.savefig(base_dir / f'{slide_path.stem}_top{k:03}_all.png', dpi=300)
# %%
# plot every lth tile from top k tiles
ind = every_l

rows, columns, img_size = 2, 5, 5
plt.figure(figsize=(columns * img_size, rows * (img_size + 0.5)))
for i in range(len(ind)):
    x, y = coords[ind[i]]
    margin = 250
    size = 1013
    tile = slide.read_region((x - margin, y - margin), 0, (size + 2 * margin, size + 2 * margin))

    plt.subplot(rows, columns, i + 1)
    plt.imshow(tile)
    plt.title(i)

    box_x = margin  # x-coordinate of the top-left corner of the rectangle
    box_y = margin  # y-coordinate of the top-left corner of the rectangle
    box_width = size  # Width of the rectangle
    box_height = size  # Height of the rectangle

    rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    plt.axis('off')
plt.tight_layout()
plt.savefig(base_dir / 'tiles' / f'{slide_path.stem}_top{k:03}_every{l:02}.png', dpi=300)
# %%
# plot top and bottom scored tiles from top k tiles
l = 2
values_scores_high, indices_scores_high = scores[indices].topk(l)
print(values_scores_high, indices_scores_high)
values_scores_low, indices_scores_low = scores[indices].topk(l, largest=False)
print(values_scores_low, indices_scores_low)
plot_indices = [indices[i] for i in [*indices_scores_high, *indices_scores_low]]
print(plot_indices)

rows, columns, img_size = 2, 2, 5
plt.figure(figsize=(columns * img_size, rows * (img_size + 0.5)))
for i in range(len(plot_indices)):
    print(scores[plot_indices[i]])
    x, y = coords[plot_indices[i]]
    margin = 250
    size = 1013
    tile = slide.read_region((x - margin, y - margin), 0, (size + 2 * margin, size + 2 * margin))

    plt.subplot(rows, columns, i + 1)
    plt.imshow(tile)
    plt.title(i)

    box_x = margin  # x-coordinate of the top-left corner of the rectangle
    box_y = margin  # y-coordinate of the top-left corner of the rectangle
    box_width = size  # Width of the rectangle
    box_height = size  # Height of the rectangle

    rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    plt.axis('off')
plt.tight_layout()
# plt.savefig(base_dir / 'tiles' / f'{slide_path.stem}_top{k:03}_every{l:02}.png', dpi=300)
# %%
plot_scores(coords, rollout, image, indices=plot_indices)
plt.show()
# %%
plot_indices
# %%
scores[indices].min(), scores[indices].max()
# %%
scores.min(), scores.max()
# %%
indices.shape
# %% md
### Save images for user study for every whole slide image
# %%
k = 50  # top k tiles
l = 10  # every lth tile

for s in tqdm(slides):
    # if slides.index(s) in [28, 29, 35]:  # 29 is completely blurry
    #     continue
    if Path(base_dir / 'tiles' / f'{s.stem}_tiles_top{k:03}_every{l:02}.png').exists():
        continue
    feature_path = feature_dir / f'{s.stem}.h5'

    h5_file = h5py.File(feature_path)
    features = torch.Tensor(np.array(h5_file['feats'])).unsqueeze(0)
    coords = torch.Tensor(np.array(h5_file['coords']))
    coords = [(coords[i, 0].int().item(), coords[i, 1].int().item()) for i in range(coords.shape[0])]

    slide = openslide.OpenSlide(s)
    level = len(slide.level_downsamples) - 1
    d = slide.level_downsamples[level]
    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))

    rollout = generate_rollout(classifier.model, features, start_layer=0).squeeze(0)

    plot_scores(coords, rollout, image, overlay=False)
    plt.savefig(base_dir / 'heatmaps' / f'{s.stem}_attention.png', dpi=300)
    plt.show()

    values, indices = rollout.topk(k)
    every_l = [indices[i] for i in range(k) if i % l == 0]

    plot_scores(coords, rollout, image, indices=indices)
    plt.savefig(base_dir / 'heatmaps' / f'{s.stem}_attention_top{k:03}.png', dpi=300)
    plt.show()

    plot_scores(coords, rollout, image, indices=every_l)
    plt.savefig(base_dir / 'heatmaps' / f'{s.stem}_attention_top{k:03}_every{l:02}.png', dpi=300)
    plt.show()

    # plot top tiles
    ind = every_l

    rows, columns, img_size = 1, 5, 5
    plt.figure(figsize=(columns * img_size, rows * (img_size + 0.5)))
    for i in range(len(ind)):
        x, y = coords[ind[i]]
        margin = 250
        size = 1013
        tile = slide.read_region((x - margin, y - margin), 0, (size + 2 * margin, size + 2 * margin))

        plt.subplot(rows, columns, i + 1)
        plt.imshow(tile)
        plt.title(i)

        box_x = margin  # x-coordinate of the top-left corner of the rectangle
        box_y = margin  # y-coordinate of the top-left corner of the rectangle
        box_width = size  # Width of the rectangle
        box_height = size  # Height of the rectangle

        rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black',
                                 facecolor='none')
        plt.gca().add_patch(rect)

        plt.axis('off')
    plt.tight_layout()
    plt.savefig(base_dir / 'tiles' / f'{s.stem}_tiles_top{k:03}_every{l:02}.png', dpi=300)
    plt.show()

# %%
k = 50  # top k tiles
l = 10  # every lth tile

for s in tqdm(slides):
    if Path(base_dir / 'tiles_cls_scores' / f'{s.stem}_tiles_top{k:03}_high{l:02}_low{l:02}_cls_scores.png').exists():
        continue
    feature_path = feature_dir / f'{s.stem}.h5'

    h5_file = h5py.File(feature_path)
    features = torch.Tensor(np.array(h5_file['feats'])).unsqueeze(0)
    coords = torch.Tensor(np.array(h5_file['coords']))
    coords = [(coords[i, 0].int().item(), coords[i, 1].int().item()) for i in range(coords.shape[0])]

    slide = openslide.OpenSlide(s)
    level = len(slide.level_downsamples) - 1
    d = slide.level_downsamples[level]
    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = np.array(image.convert("RGB"))

    rollout = generate_rollout(classifier.model, features, start_layer=0).squeeze(0)

    # plot_scores(coords, rollout, image, overlay=False)
    # plt.savefig(base_dir / 'heatmaps' / f'{s.stem}_attention.png', dpi=300)
    # plt.show()

    values, indices = rollout.topk(k)
    every_l = [indices[i] for i in range(k) if i % l == 0]

    # compute classificaiton scores
    n = features.shape[1]
    scores = torch.zeros(n)
    for i in tqdm(range(n)):
        out = classifier.model(features[:, i:i + 1, :]).squeeze(0)
        scores[i] = torch.sigmoid(out)

    l = 2
    values_scores_high, indices_scores_high = scores[indices].topk(l)
    print(values_scores_high, indices_scores_high, values_scores_low, indices_scores_low, plot_indices)
    values_scores_low, indices_scores_low = scores[indices].topk(l, largest=False)
    plot_indices = [indices[i] for i in [*indices_scores_high, *indices_scores_low]]

    plot_scores(coords, rollout, image, indices=plot_indices)
    plt.savefig(base_dir / 'heatmaps' / f'{s.stem}_attention_top{l:03}_bottom{l:03}_cls_scores.png', dpi=300)
    plt.show()

    rows, columns, img_size = 2, 2, 5
    plt.figure(figsize=(columns * img_size, rows * (img_size + 0.5)))
    for i in range(len(plot_indices)):
        x, y = coords[plot_indices[i]]
        margin = 250
        size = 1013
        tile = slide.read_region((x - margin, y - margin), 0, (size + 2 * margin, size + 2 * margin))

        plt.subplot(rows, columns, i + 1)
        plt.imshow(tile)
        plt.title(i)

        box_x = margin  # x-coordinate of the top-left corner of the rectangle
        box_y = margin  # y-coordinate of the top-left corner of the rectangle
        box_width = size  # Width of the rectangle
        box_height = size  # Height of the rectangle

        rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black',
                                 facecolor='none')
        plt.gca().add_patch(rect)

        plt.axis('off')
    plt.tight_layout()
    plt.savefig(base_dir / 'tiles_cls_scores' / f'{s.stem}_tiles_top{k:03}_high{l:02}_low{l:02}_cls_scores.png',
                dpi=300)
    plt.show()

# %% md
## Figures for visualization of MSI transformer paper
# %%
figure_path = Path('/Users/sophia.wagner/Documents/PhD/projects/2022_MSI_transformer/figures/attention_visualization')
# %%
# save original image
plt.figure(figsize=(15, 15))
plt.imshow(image)
plt.axis('off')
plt.savefig(figure_path / f'{slide_path.stem}.png', dpi=300, bbox_inches='tight', pad_inches=0)
# %% md
### plot attention maps
# %%
rollout = generate_rollout(classifier.model, features, start_layer=0).squeeze(0)
# %%
# save the attention map
plt.figure(figsize=(6, 6))
plot_scores(coords, rollout, image, overlay=True, colormap='viridis', crop=True)
plt.savefig(figure_path / f'{slide_path.stem}_{target}_attention.svg', format='svg', bbox_inches='tight', pad_inches=0)
plt.show()
# %% md
### plot class scores
# %%
n = features.shape[1]
scores = np.zeros(n)
for i in tqdm(range(n)):
    out = classifier.model(features[:, i:i + 1, :]).squeeze(0)
    scores[i] = torch.sigmoid(out)
# %%
plt.figure(figsize=(6, 6))
plot_scores(coords, scores, image, overlay=True, colormap='RdBu_r', clamp=False, norm=False, crop=True)
plt.savefig(figure_path / f'{slide_path.stem}_{target}_cls_scores.svg', format='svg', bbox_inches='tight', pad_inches=0)
plt.show()
# %% md
### plot attention heads
# %%
classifier.model(features)
blocks = classifier.model.transformer.layers
all_attentions_heads = []
for blk in blocks:
    attn_heads = blk[0].fn.get_attention_map()
    all_attentions_heads.append(attn_heads)
attn_heads = torch.cat(all_attentions_heads, dim=0)
print(attn_heads.shape)
# %%
layer = 0
rows, columns, img_size = 2, 8, 4
plt.figure(figsize=(columns * img_size, rows * 3))
for l in range(2):
    for i in range(columns):
        att_vis = attn_heads.clone()
        att_vis = att_vis[l][i][0, 1:]  # [1:, 0], ([0, 1:] is correct)

        plt.subplot(rows, columns, l * columns + i + 1)
        plot_scores(coords, att_vis, image, clamp=0.05, overlay=False, colormap='viridis', crop=True)
        plt.axis('off')
plt.tight_layout()
# plt.savefig(figure_path / f'{slide_path.stem}_attention_heads.svg', format='svg', bbox_inches = 'tight', pad_inches = 0)
plt.show()
# %% md
### plot color bars
# %%
import matplotlib.pyplot as plt

# Create a ScalarMappable object with the "viridis" colormap
sm = plt.cm.ScalarMappable(cmap="viridis")

# Set the colorbar limits and labels
# cbar = 
plt.colorbar(sm, orientation='horizontal', ticks=[0, 1])
# cbar.ax.set_xticklabels(['0', '1'])

# Show the plot
plt.savefig(figure_path / f'viridis_colorbar.svg', format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
import matplotlib.pyplot as plt

# Create a ScalarMappable object with the "viridis" colormap
sm = plt.cm.ScalarMappable(cmap="RdBu_r")

# Set the colorbar limits and labels
# cbar = 
cbar = plt.colorbar(sm, orientation='horizontal', ticks=[0, 1])
cbar.ax.set_xticklabels(['MSS', 'MSI-high'])

# Show the plot
plt.savefig(figure_path / f'RdBu_colorbar.svg', format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
import matplotlib.pyplot as plt

# Create a ScalarMappable object with the "viridis" colormap
sm = plt.cm.ScalarMappable(cmap="viridis")

# Set the colorbar limits and labels
cbar = plt.colorbar(sm, ticks=[0, 1])
cbar.ax.set_yticklabels(['0', '1'])

# Remove the surrounding frame and axes
cbar.outline.set_visible(False)
cbar.ax.axis('off')

# Set the figure size to only display the colorbar
fig = plt.gcf()
fig.set_size_inches(2, 6)  # Adjust the size as needed

# Show the colorbar
plt.show()

# %% md

# %%
n = features.shape[1]
scores = np.zeros(n)
for i in tqdm(range(n)):
    out = classifier.model(features[:, i:i + 1, :]).squeeze(0)
    scores[i] = torch.sigmoid(out)
# %%
plot_scores(coords, scores, image, overlay=True, colormap='RdBu_r', clamp=False, norm=False)
plt.show()
