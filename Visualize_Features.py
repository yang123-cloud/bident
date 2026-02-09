# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:24:02 2026

@author: yangzhibo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE 可视化：时域(temporal) + 小波(contextual) 特征融合
输入：data_dir 下应包含 temporal.npy, contextual.npy, labels.npy, 可选 label_map.json
输出：2D t-SNE 散点图（PNG）并保存
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Linux Libertine O']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Linux Libertine O'
rcParams['mathtext.it'] = 'Linux Libertine O:italic'
rcParams['mathtext.bf'] = 'Linux Libertine O:bold'

def load_data(data_dir):
    temporal_path = os.path.join(data_dir, 'temporal.npy')
    contextual_path = os.path.join(data_dir, 'contextual.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    labelmap_path = os.path.join(data_dir, 'label_map.json')

    if not os.path.exists(temporal_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"需要 temporal.npy 和 labels.npy 在 {data_dir}")

    temporal = np.load(temporal_path)  # (N, T, P)
    labels = np.load(labels_path)      # (N,)
    contextual = None
    if os.path.exists(contextual_path):
        contextual = np.load(contextual_path)  # (N, 3, F, T) or maybe (N,3,agg,agg)
    label_map = None
    if os.path.exists(labelmap_path):
        with open(labelmap_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
    return temporal, contextual, labels, label_map

def prepare_feature_vectors(temporal, contextual, flatten_contextual=True):
    """
    把 temporal/contextual 转为向量：
      - temporal: flatten (T * P)
      - contextual: 若存在则 flatten (3 * F * T) 或对每通道做 PCA 后合并
    返回 fused_vectors (N, D_temporal + D_contextual)
    """
    N = temporal.shape[0]
    X_tem = temporal.reshape(N, -1).astype(np.float32)  # (N, T*P)
    # 标准化 temporal
    scaler_tem = StandardScaler()
    X_tem = scaler_tem.fit_transform(X_tem)

    if contextual is None or contextual.size == 0:
        return X_tem, 'temporal_only'
    # contextual shape expected (N, C, F, T)
    X_ctx = contextual.reshape(N, -1).astype(np.float32)

    # 标准化 contextual
    scaler_ctx = StandardScaler()
    X_ctx = scaler_ctx.fit_transform(X_ctx)

    # 默认直接拼接：若维度过高，后续 PCA 会再降
    X_fused = np.concatenate([X_tem, X_ctx], axis=1)
    return X_fused, 'temporal_plus_contextual'

def reduce_with_pca(X, n_components=50, random_state=42):
    """
    如果维度 >> n_components，先用 PCA 降到 n_components（更快且去噪）
    返回降维后的 X_pca
    """
    n_samples, n_feat = X.shape
    n_comp = min(n_components, n_samples - 1, n_feat)
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def run_tsne(X, n_components=2, perplexity=30, random_state=42, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, random_state=random_state, init='pca')
    X_emb = tsne.fit_transform(X)
    return X_emb

def plot_embedding(X_emb, labels, label_map=None, out_png='tsne.pdf',
                   figsize=(12,12),
                   marker_size=450,        # scatter 点的大小（s，点面积，单位 pts^2）
                   title_size=45,
                   label_size=38,
                   tick_size=38,
                   legend_fontsize=33,
                   legend_markerscale=1.0,
                   dpi=300):
    """
    X_emb: (N,2)
    labels: (N,) integer labels
    label_map: dict label->idx mapping from file, but we want idx->name mapping. If provided as {name:idx}, invert it.
    参数说明：marker_size 是传给 scatter 的 s（点面积，默认 80），
            title_size/label_size/tick_size/legend_fontsize 可放大字体。
    """
    import matplotlib.pyplot as plt
    unique = np.unique(labels)
    n_classes = unique.size

    # 建 readable names
    if label_map is None:
        names = {int(c): str(int(c)) for c in unique}
    else:
        try:
            inv = {int(v): k for k, v in label_map.items()}
            names = {int(c): inv.get(int(c), str(int(c))) for c in unique}
        except Exception:
            names = {int(c): str(int(c)) for c in unique}

    # 全局样式微调（不会影响其他代码太多）
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 如果你想给每类一个固定颜色，matplotlib 会自动分配颜色循环
    for lab in unique:
        sel = labels == lab
        # s 参数接受点面积（pts^2），edgecolors='w' 给点加白色边框，linewidths 让点更清晰
        ax.scatter(X_emb[sel, 0], X_emb[sel, 1],
                   label=names[int(lab)],
                   s=marker_size,
                   alpha=0.85,
                   edgecolors='w',
                   linewidths=0.5)

    # 字体大小
    # ax.set_xlabel('t-SNE 1', fontsize=label_size)
    # ax.set_ylabel('t-SNE 2', fontsize=label_size)
    ax.set_title('(b) Visualization of USTC-TFC2016', fontsize=title_size,fontweight='bold', pad=12, y=-0.12)
    #(b) Visualization of USTC-TFC2016 #(a) Visualization of IDS2017&2018
    # 刻度字体
    # ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # 图例：markerscale 放大图例中的marker；frameon/edgecolor 控制图例外观
   
    
    legend = ax.legend(loc='lower left', fontsize=legend_fontsize, markerscale=legend_markerscale,ncol=1,
                       columnspacing=0,   # ↓ 列与列之间的水平间距（默认约 2.0）
                        labelspacing=0,    # ↓ 不同行之间的垂直间距（默认约 0.5~0.8）
                        handletextpad=0,   # ↓ marker 与文字之间的距离
                        handlelength=0.6,   # 关键：缩短 marker 句柄区域
                        borderpad=0.1,      # 同时缩小 legend 内边距
                        frameon=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    # 可选地增加图例项之间的间距和标记与文本的距离
    legend._legend_box.align = "left"

    # 网格与布局
    # ax.grid(True, alpha=0.25, linestyle='--')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    # 保存为矢量 PDF 或高分辨率 PNG
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./multi_result/USTC-TFC2016_result', help='包含 temporal.npy, contextual.npy, labels.npy 的目录')
    parser.add_argument('--out_png', type=str, default='tsne_fusion.pdf', help='输出图片路径')
    parser.add_argument('--sample_per_class', type=int, default=100, help='若样本很多，可每类下采样到此数量')
    parser.add_argument('--pca_dim', type=int, default=32, help='PCA 维数（t-SNE 前的预降维）')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_iter', type=int, default=500)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    temporal, contextual, labels, label_map = load_data(args.data_dir)
    
    N = temporal.shape[0]
    print(f"Loaded data: temporal {temporal.shape}, contextual {None if contextual is None else contextual.shape}, labels {labels.shape}")

    # optional: 先按类别下采样以避免某一类主导可视化
    if args.sample_per_class is not None:
        unique, counts = np.unique(labels, return_counts=True)
        sel_indices = []
        rng = np.random.RandomState(args.random_state)
        for c in unique:
            idxs = np.where(labels == c)[0]
            if len(idxs) > args.sample_per_class:
                pick = rng.choice(idxs, size=args.sample_per_class, replace=False)
            else:
                pick = idxs
            sel_indices.extend(pick.tolist())
        sel_indices = np.array(sel_indices, dtype=int)
        temporal = temporal[sel_indices]
        if contextual is not None and contextual.size > 0:
            contextual = contextual[sel_indices]
        labels = labels[sel_indices]
        print(f"Subsampled to {len(sel_indices)} samples.")

    X_fused, mode = prepare_feature_vectors(temporal, contextual)
    print("Feature mode:", mode, "Feature dim:", X_fused.shape)

    # PCA 预降维（能显著加速 t-SNE）
    X_pca, pca = reduce_with_pca(X_fused, n_components=args.pca_dim, random_state=args.random_state)
    print("After PCA shape:", X_pca.shape)

    # 运行 t-SNE
    X_emb = run_tsne(X_pca, perplexity=args.tsne_perplexity, n_iter=args.tsne_iter, random_state=args.random_state)
    print("t-SNE finished.")
    
    if label_map is not None:
        # 兼容 label_map 可能是 {'Benign': 0, ...} 或者键为数字的情况
        old_name = 'Benign'
        new_name = '360'
        if old_name in label_map:
            # 直接把键改成 new_name
            value = label_map.pop(old_name)
            label_map[new_name] = value
        else:
            # 如果 label_map 的键是数字但对应的字符串等于 old_name（小概率），也尝试匹配
            for k, v in list(label_map.items()):
                if str(k) == old_name:
                    value = label_map.pop(k)
                    label_map[new_name] = value
                    break

    # 画图并保存
    plot_embedding(X_emb, labels, label_map=label_map, out_png=args.out_png)

if __name__ == '__main__':
    main()


