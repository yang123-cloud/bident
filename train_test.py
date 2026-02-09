import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_mutual_info_score

from datasets import TrafficScopeDataset
from models import TrafficScope, TrafficScopeTemporal, TrafficScopeContextual
from metaconst import TRAFFIC_SCOPE, TRAFFIC_SCOPE_TEMPORAL, TRAFFIC_SCOPE_CONTEXTUAL
from interpretability import attention_rollout, attention_normalize
from robustness import get_robustness_test_dataset

import torch.serialization

# def train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                        temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                        use_temporal, use_contextual,
#                        num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, device):
#     train_dataset = TrafficScopeDataset(data_dir, agg_scales, train_idx)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     if use_temporal and use_contextual:
#         model = TrafficScope(temporal_seq_len, packet_len,
#                              freqs_size, agg_scale_num, agg_points_num,
#                              num_heads, num_layers, num_classes, dropout)
#     elif use_temporal and not use_contextual:
#         model = TrafficScopeTemporal(temporal_seq_len, packet_len,
#                                      num_heads, num_layers, num_classes, dropout)
#     elif not use_temporal and use_contextual:
#         model = TrafficScopeContextual(agg_scale_num, agg_points_num, freqs_size,
#                                        num_heads, num_layers, num_classes, dropout)
#     else:
#         print('should specify at least one input type')
#         return
#     # 取消注释权重初始化代码
#     def init_weights(m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
    
#     model.apply(init_weights)
    
#     model.to(device)
    
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    

#     print('load model successfully. Start training...')
#     train_start_time = time.time()
#     for epoch in range(epochs):
#         print(f'\nEpoch {epoch+1}\n--------------------')
#         epoch_start_time = time.time()
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(train_dataloader):
#             batch_start_time = time.time()
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
#                 probs = model(batch_contextual_data, batch_contextual_segments)

#             loss = loss_fn(probs, batch_labels)
#             # 计算准确率
#             preds = probs.argmax(1)
#             acc = (preds == batch_labels).float().mean()
            
#             loss.backward()
#             # 添加梯度裁剪
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
#             optimizer.step()
#             optimizer.zero_grad()
#             batch_end_time = time.time()

#             if batch_idx % 100 == 0:
#                 print(f'loss: {loss.item()}, acc: {acc.item()}, time cost: {batch_end_time - batch_start_time} s, '
#                       f'[{batch_idx + 1}]/[{len(train_dataloader)}]')
                
#         epoch_end_time = time.time()
#         print(f'Epoch time cost: {epoch_end_time - epoch_start_time} s')

#     train_end_time = time.time()
#     model_dir = os.path.split(model_path)[0]
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     torch.save(model, model_path)
#     print(f'save {model_path} successfully')
#     print(f'train {model.model_name} Done! Time cost: {train_end_time - train_start_time}')


# def test_TrafficScope(data_dir, agg_scales, test_idx,
#                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                       batch_size, model_path, num_classes, result_path, device,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
#     test_dataset = TrafficScopeDataset(data_dir, agg_scales, test_idx)
#     if robust_test_name:
#         test_dataset = get_robustness_test_dataset(test_dataset,
#                                                    robust_test_name,
#                                                    rho, kappa, different, alpha, eta, beta, gamma)
#         print(f'generate robust test dataset with {robust_test_name} successfully')
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     data_num = len(test_dataset)

#     if not os.path.exists(model_path):
#         print('model path does not exist')
#         return
    
    
#     model = torch.load(model_path)
#     model.eval()

#     print('load model successfully. Start testing...')
#     loss_fn = nn.CrossEntropyLoss()
#     test_loss = 0

#     y_preds = torch.zeros(data_num)
#     y_true = torch.zeros(data_num)
#     y_probs = torch.zeros((data_num, num_classes))
#     # used for saving attention weights
#     temporal_attention_masks = torch.zeros((data_num, temporal_seq_len, temporal_seq_len))
#     contextual_attention_masks = torch.zeros((data_num, agg_scale_num*agg_points_num, agg_scale_num*agg_points_num))
#     fusion_attention_masks = torch.zeros((data_num, temporal_seq_len, agg_scale_num*agg_points_num))
#     # used for saving latent features
#     temporal_features = torch.zeros((data_num, temporal_seq_len, packet_len))
#     contextual_features = torch.zeros((data_num, agg_scale_num*agg_points_num, freqs_size))
#     fusion_futures = torch.zeros((data_num, temporal_seq_len, packet_len))

#     data_idx = 0
#     test_start_time = time.time()
#     with torch.no_grad():
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(test_dataloader):
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#                 batch_temporal_attention_masks = attention_rollout(model.get_temporal_attention_weights(),
#                                                                    discard_ratio=0.3)
#                 batch_contextual_attention_masks = attention_rollout(model.get_contextual_attention_weights(),
#                                                                      discard_ratio=0.3)
#                 batch_fusion_attention_masks = attention_normalize(model.get_fusion_attention_weights(),
#                                                                    discard_ratio=0.3)
#                 batch_temporal_features = model.get_temporal_features()
#                 batch_contextual_features = model.get_contextual_features()
#                 batch_fusion_features = model.get_fusion_features()
#                 # print(len(model.temporal_encoder.attention_weights))
#                 # print(model.temporal_encoder.attention_weights[0].shape)
#                 # print(model.contextual_encoder.attention_weights[0].shape)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
#                 probs = model(batch_contextual_data, batch_contextual_segments)

#             test_loss += loss_fn(probs, batch_labels).item()
#             preds = probs.argmax(1)
#             batch_len = probs.size(0)
#             y_preds[data_idx:data_idx + batch_len] = preds.cpu()
#             y_probs[data_idx:data_idx + batch_len] = probs.cpu()
#             y_true[data_idx:data_idx + batch_len] = batch_labels.cpu()

#             if model.model_name == TRAFFIC_SCOPE:
#                 temporal_attention_masks[data_idx:data_idx+batch_len] = batch_temporal_attention_masks.cpu()
#                 contextual_attention_masks[data_idx:data_idx+batch_len] = batch_contextual_attention_masks.cpu()
#                 fusion_attention_masks[data_idx:data_idx+batch_len] = batch_fusion_attention_masks.cpu()

#                 temporal_features[data_idx:data_idx+batch_len] = batch_temporal_features.cpu()
#                 contextual_features[data_idx:data_idx+batch_len] = batch_contextual_features.cpu()
#                 fusion_futures[data_idx:data_idx+batch_len] = batch_fusion_features.cpu()
#             data_idx += batch_len
    
#     acc = accuracy_score(y_true.numpy(), y_preds.numpy())
#     pre = precision_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     rec = recall_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     f1 = f1_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     print(f'\nTest loss: {test_loss / len(test_dataloader)}\n'
#           f'Acc: {acc} Pre: {pre} Rec: {rec} F1: {f1}\n'
#           f'time cost: {time.time() - test_start_time}')
#     print(f'test {model.model_name} Done!')
#     # save results
#     result_dir, result_name = os.path.split(result_path)
#     y_true_name = os.path.splitext(result_name)[0] + '_y_true.npy'
#     y_preds_name = os.path.splitext(result_name)[0] + '_y_preds.npy'
#     y_probs_name = os.path.splitext(result_name)[0] + '_y_probs.npy'
#     temporal_attention_masks_name = os.path.splitext(result_name)[0] + '_temporal_attention_masks.npy'
#     contextual_attention_masks_name = os.path.splitext(result_name)[0] + '_contextual_attention_masks.npy'
#     fusion_attention_masks_name = os.path.splitext(result_name)[0] + '_fusion_attention_masks.npy'
#     temporal_features_name = os.path.splitext(result_name)[0] + '_temporal_features.npy'
#     contextual_features_name = os.path.splitext(result_name)[0] + '_contextual_features.npy'
#     fusion_features_name = os.path.splitext(result_name)[0] + '_fusion_features.npy'

#     y_true_path = os.path.join(result_dir, y_true_name)
#     y_preds_path = os.path.join(result_dir, y_preds_name)
#     y_probs_path = os.path.join(result_dir, y_probs_name)
#     temporal_attention_masks_path = os.path.join(result_dir, temporal_attention_masks_name)
#     contextual_attention_masks_path = os.path.join(result_dir, contextual_attention_masks_name)
#     fusion_attention_masks_path = os.path.join(result_dir, fusion_attention_masks_name)
#     temporal_features_path = os.path.join(result_dir, temporal_features_name)
#     contextual_features_path = os.path.join(result_dir, contextual_features_name)
#     fusion_features_path = os.path.join(result_dir, fusion_features_name)
    
#     #电脑D盘内存不足，无法保存
#     # if not os.path.exists(result_dir):
#     #     os.makedirs(result_dir)
#     # np.save(y_true_path, y_true.numpy())
#     # np.save(y_preds_path, y_preds.numpy())
#     # np.save(y_probs_path, y_probs.numpy())
#     # print(f'save {y_true_path}, {y_preds_path}, {y_probs_path} successfully')
#     # if model.model_name == TRAFFIC_SCOPE:
#     #     np.save(temporal_attention_masks_path, temporal_attention_masks.numpy())
#     #     np.save(contextual_attention_masks_path, contextual_attention_masks.numpy())
#     #     np.save(fusion_attention_masks_path, fusion_attention_masks.numpy())
#     #     np.save(temporal_features_path, temporal_features.numpy())
#     #     np.save(contextual_features_path, contextual_features.numpy())
#     #     np.save(fusion_features_path, fusion_futures.numpy())
#     #     print(f'save {temporal_attention_masks_path}, {contextual_attention_masks_path}, '
#     #           f'{fusion_attention_masks_path}, '
#     #           f'{temporal_features_path}, {contextual_features_path}, {fusion_features_path} successfully')


# def train_test_helper(data_dir, agg_scales, model_name, agg_scale_num, agg_points_num, batch_size,
#                       temporal_seq_len, packet_len, freqs_size, use_temporal, use_contextual, is_train, is_test,
#                       num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, result_path,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None,
#                       k_fold=None, gpu_id=0):
#     os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu_id)
#     dataset = TrafficScopeDataset(data_dir, agg_scales)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if k_fold:
#         skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
#         for train_idx, test_idx in skf.split(dataset.temporal_data.numpy(), dataset.labels.numpy()):
#             if model_name == TRAFFIC_SCOPE:
#                 if is_train:
#                     train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                                        temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                        use_temporal, use_contextual,
#                                        num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
#                                        device)
#                 if is_test:
#                     test_TrafficScope(data_dir, agg_scales, test_idx,
#                                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                       batch_size, model_path, num_classes, result_path, device,
#                                       robust_test_name,
#                                       rho, kappa, different, alpha, eta, beta, gamma)
#     else:
#         indices = np.arange(dataset.temporal_data.shape[0])
#         train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
#         if model_name == TRAFFIC_SCOPE:
#             if is_train:
#                 train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                                    temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                    use_temporal, use_contextual,
#                                    num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
#                                    device)
#             if is_test:
#                 test_TrafficScope(data_dir, agg_scales, test_idx,
#                                   temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                   batch_size, model_path, num_classes, result_path, device,
#                                   robust_test_name,
#                                   rho, kappa, different, alpha, eta, beta, gamma)
# 之前为正常的主训练分类代码
######################################################################################

# def train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                         temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                         use_temporal, use_contextual,
#                         num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, device):
#     train_dataset = TrafficScopeDataset(data_dir, agg_scales, train_idx)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     if use_temporal and use_contextual:
#         model = TrafficScope(temporal_seq_len, packet_len,
#                               freqs_size, agg_scale_num, agg_points_num,
#                               num_heads, num_layers, num_classes, dropout)
#     elif use_temporal and not use_contextual:
#         model = TrafficScopeTemporal(temporal_seq_len, packet_len,
#                                       num_heads, num_layers, num_classes, dropout)
#     elif not use_temporal and use_contextual:
#         model = TrafficScopeContextual(agg_scale_num, agg_points_num, freqs_size,
#                                         num_heads, num_layers, num_classes, dropout)
#     else:
#         print('should specify at least one input type')
#         return
#     # 取消注释权重初始化代码
#     def init_weights(m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
    
#     model.apply(init_weights)
    
#     model.to(device)
    
#     ##################################损失函数计算
#     loss_fn = nn.MSELoss()#CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    

#     print('load model successfully. Start training...')
#     train_start_time = time.time()
#     for epoch in range(epochs):
#         print(f'\nEpoch {epoch+1}\n--------------------')
#         epoch_start_time = time.time()
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(train_dataloader):
#             batch_start_time = time.time()
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
#                 probs = model(batch_contextual_data, batch_contextual_segments)
#             #计算每一部分的自编码损失
#             loss = loss_fn(probs[0], batch_temporal_data) + loss_fn(probs[1], batch_contextual_data)
#             #loss_fn(probs, batch_labels)
            
#             loss.backward()
#             # 添加梯度裁剪
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
#             optimizer.step()
#             optimizer.zero_grad()
#             batch_end_time = time.time()

#             if batch_idx % 100 == 0:
#                 print(f'loss: {loss.item()}, time cost: {batch_end_time - batch_start_time} s, '
#                       f'[{batch_idx + 1}]/[{len(train_dataloader)}]')
                
#         epoch_end_time = time.time()
#         print(f'Epoch time cost: {epoch_end_time - epoch_start_time} s')

#     train_end_time = time.time()
#     model_dir = os.path.split(model_path)[0]
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     torch.save(model, model_path)
#     print(f'save {model_path} successfully')
#     print(f'train {model.model_name} Done! Time cost: {train_end_time - train_start_time}')


# def test_TrafficScope(data_dir, agg_scales, test_idx,
#                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                       batch_size, model_path, num_classes, result_path, device,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
#     test_dataset = TrafficScopeDataset(data_dir, agg_scales, test_idx)
#     if robust_test_name:
#         test_dataset = get_robustness_test_dataset(test_dataset,
#                                                     robust_test_name,
#                                                     rho, kappa, different, alpha, eta, beta, gamma)
#         print(f'generate robust test dataset with {robust_test_name} successfully')
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     data_num = len(test_dataset)

#     if not os.path.exists(model_path):
#         print('model path does not exist')
#         return
    
    
#     model = torch.load(model_path)
#     model.eval()

#     print('load model successfully. Start testing...')
    
#     ##################################损失函数计算
#     loss_fn = nn.MSELoss()#CrossEntropyLoss()
#     test_loss = 0

#     y_preds = torch.zeros(data_num)
#     y_true = torch.zeros(data_num)
#     y_probs = torch.zeros((data_num, num_classes))
#     # used for saving attention weights
#     temporal_attention_masks = torch.zeros((data_num, temporal_seq_len, temporal_seq_len))
#     contextual_attention_masks = torch.zeros((data_num, agg_scale_num*agg_points_num, agg_scale_num*agg_points_num))
#     fusion_attention_masks = torch.zeros((data_num, temporal_seq_len, agg_scale_num*agg_points_num))
#     # used for saving latent features
#     temporal_features = torch.zeros((data_num, temporal_seq_len, packet_len))
#     contextual_features = torch.zeros((data_num, agg_scale_num*agg_points_num, freqs_size))
#     fusion_futures = torch.zeros((data_num, temporal_seq_len, packet_len))

#     data_idx = 0
#     test_start_time = time.time()
#     with torch.no_grad():
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(test_dataloader):
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#                 batch_temporal_attention_masks = attention_rollout(model.get_temporal_attention_weights(),
#                                                                     discard_ratio=0.3)
#                 batch_contextual_attention_masks = attention_rollout(model.get_contextual_attention_weights(),
#                                                                       discard_ratio=0.3)
#                 batch_fusion_attention_masks = attention_normalize(model.get_fusion_attention_weights(),
#                                                                     discard_ratio=0.3)
#                 batch_temporal_features = model.get_temporal_features()
#                 batch_contextual_features = model.get_contextual_features()
#                 batch_fusion_features = model.get_fusion_features()
#                 # print(len(model.temporal_encoder.attention_weights))
#                 # print(model.temporal_encoder.attention_weights[0].shape)
#                 # print(model.contextual_encoder.attention_weights[0].shape)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
#                 probs = model(batch_contextual_data, batch_contextual_segments)
            
#             ############################################计算每一部分的自编码损失
#             test_loss += loss_fn(probs[0], batch_temporal_data).item() + loss_fn(probs[1], batch_contextual_data).item()
#             # test_loss += loss_fn(probs, batch_labels).item()
#             # preds = probs.argmax(1)
#             # batch_len = probs.size(0)
#             # y_preds[data_idx:data_idx + batch_len] = preds.cpu()
#             # y_probs[data_idx:data_idx + batch_len] = probs.cpu()
#             # y_true[data_idx:data_idx + batch_len] = batch_labels.cpu()

#             # if model.model_name == TRAFFIC_SCOPE:
#             #     temporal_attention_masks[data_idx:data_idx+batch_len] = batch_temporal_attention_masks.cpu()
#             #     contextual_attention_masks[data_idx:data_idx+batch_len] = batch_contextual_attention_masks.cpu()
#             #     fusion_attention_masks[data_idx:data_idx+batch_len] = batch_fusion_attention_masks.cpu()

#             #     temporal_features[data_idx:data_idx+batch_len] = batch_temporal_features.cpu()
#             #     contextual_features[data_idx:data_idx+batch_len] = batch_contextual_features.cpu()
#             #     fusion_futures[data_idx:data_idx+batch_len] = batch_fusion_features.cpu()
#             # data_idx += batch_len
    
#     # acc = accuracy_score(y_true.numpy(), y_preds.numpy())
#     # pre = precision_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     # rec = recall_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     # f1 = f1_score(y_true.numpy(), y_preds.numpy(), average='macro')
#     print(f'\nTest loss: {test_loss / len(test_dataloader)}\n'
#           f'time cost: {time.time() - test_start_time}')
#     print(f'test {model.model_name} Done!')
#     # save results
#     result_dir, result_name = os.path.split(result_path)
#     y_true_name = os.path.splitext(result_name)[0] + '_y_true.npy'
#     y_preds_name = os.path.splitext(result_name)[0] + '_y_preds.npy'
#     y_probs_name = os.path.splitext(result_name)[0] + '_y_probs.npy'
#     temporal_attention_masks_name = os.path.splitext(result_name)[0] + '_temporal_attention_masks.npy'
#     contextual_attention_masks_name = os.path.splitext(result_name)[0] + '_contextual_attention_masks.npy'
#     fusion_attention_masks_name = os.path.splitext(result_name)[0] + '_fusion_attention_masks.npy'
#     temporal_features_name = os.path.splitext(result_name)[0] + '_temporal_features.npy'
#     contextual_features_name = os.path.splitext(result_name)[0] + '_contextual_features.npy'
#     fusion_features_name = os.path.splitext(result_name)[0] + '_fusion_features.npy'

#     y_true_path = os.path.join(result_dir, y_true_name)
#     y_preds_path = os.path.join(result_dir, y_preds_name)
#     y_probs_path = os.path.join(result_dir, y_probs_name)
#     temporal_attention_masks_path = os.path.join(result_dir, temporal_attention_masks_name)
#     contextual_attention_masks_path = os.path.join(result_dir, contextual_attention_masks_name)
#     fusion_attention_masks_path = os.path.join(result_dir, fusion_attention_masks_name)
#     temporal_features_path = os.path.join(result_dir, temporal_features_name)
#     contextual_features_path = os.path.join(result_dir, contextual_features_name)
#     fusion_features_path = os.path.join(result_dir, fusion_features_name)
    
#     #电脑D盘内存不足，无法保存
#     # if not os.path.exists(result_dir):
#     #     os.makedirs(result_dir)
#     # np.save(y_true_path, y_true.numpy())
#     # np.save(y_preds_path, y_preds.numpy())
#     # np.save(y_probs_path, y_probs.numpy())
#     # print(f'save {y_true_path}, {y_preds_path}, {y_probs_path} successfully')
#     # if model.model_name == TRAFFIC_SCOPE:
#     #     np.save(temporal_attention_masks_path, temporal_attention_masks.numpy())
#     #     np.save(contextual_attention_masks_path, contextual_attention_masks.numpy())
#     #     np.save(fusion_attention_masks_path, fusion_attention_masks.numpy())
#     #     np.save(temporal_features_path, temporal_features.numpy())
#     #     np.save(contextual_features_path, contextual_features.numpy())
#     #     np.save(fusion_features_path, fusion_futures.numpy())
#     #     print(f'save {temporal_attention_masks_path}, {contextual_attention_masks_path}, '
#     #           f'{fusion_attention_masks_path}, '
#     #           f'{temporal_features_path}, {contextual_features_path}, {fusion_features_path} successfully')


# def train_test_helper(data_dir, agg_scales, model_name, agg_scale_num, agg_points_num, batch_size,
#                       temporal_seq_len, packet_len, freqs_size, use_temporal, use_contextual, is_train, is_test,
#                       num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, result_path,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None,
#                       k_fold=None, gpu_id=0):
#     os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu_id)
#     dataset = TrafficScopeDataset(data_dir, agg_scales)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if k_fold:
#         skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
#         for train_idx, test_idx in skf.split(dataset.temporal_data.numpy(), dataset.labels.numpy()):
#             if model_name == TRAFFIC_SCOPE:
#                 if is_train:
#                     train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                                         temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                         use_temporal, use_contextual,
#                                         num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
#                                         device)
#                 if is_test:
#                     test_TrafficScope(data_dir, agg_scales, test_idx,
#                                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                       batch_size, model_path, num_classes, result_path, device,
#                                       robust_test_name,
#                                       rho, kappa, different, alpha, eta, beta, gamma)
#     else:
#         indices = np.arange(dataset.temporal_data.shape[0])
#         train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
#         if model_name == TRAFFIC_SCOPE:
#             if is_train:
#                 train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                                     temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                     use_temporal, use_contextual,
#                                     num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
#                                     device)
#             if is_test:
#                 test_TrafficScope(data_dir, agg_scales, test_idx,
#                                   temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                   batch_size, model_path, num_classes, result_path, device,
#                                   robust_test_name,
#                                   rho, kappa, different, alpha, eta, beta, gamma)
# 之前为自编码器的训练
####################################################################################
# import pandas as pd
# from collections import Counter
# from evt import SPOT
# import torch.nn.functional as F

# from matplotlib import pyplot as plt
# from scipy import stats

# from scipy.stats import gaussian_kde
# from scipy.signal import find_peaks
# import ruptures as rpt

# def train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
#                         temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                         use_temporal, use_contextual,
#                         num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device):

#     train_dataset = TrafficScopeDataset(data_dir, agg_scales, train_idx)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     if use_temporal and use_contextual:
#         model = TrafficScope(temporal_seq_len, packet_len,
#                               freqs_size, agg_scale_num, agg_points_num,
#                               num_heads, num_layers, num_classes, dropout)
#     elif use_temporal and not use_contextual:
#         model = TrafficScopeTemporal(temporal_seq_len, packet_len,
#                                       num_heads, num_layers, num_classes, dropout)
#     elif not use_temporal and use_contextual:
#         model = TrafficScopeContextual(agg_scale_num, agg_points_num, freqs_size,
#                                         num_heads, num_layers, num_classes, dropout)
#     else:
#         print('should specify at least one input type')
#         return
    
#     def init_weights(m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.01)
    
#     model.apply(init_weights)
#     model.to(device)
#     # 使用组合损失函数
#     def composite_loss(model_output, temporal_target, contextual_target, model_name):
#         if model_name == TRAFFIC_SCOPE:
#             temporal_recon, contextual_recon = model_output
#             temporal_loss = F.mse_loss(temporal_recon, temporal_target)
#             contextual_loss = F.mse_loss(contextual_recon, contextual_target)
#             return temporal_loss + contextual_loss, temporal_loss, contextual_loss
#         elif model_name == TRAFFIC_SCOPE_TEMPORAL:
#             temporal_recon = model_output
#             temporal_loss = F.mse_loss(temporal_recon, temporal_target)
#             return temporal_loss, temporal_loss, torch.tensor(0.0)
#         else:  # TRAFFIC_SCOPE_CONTEXTUAL
#             contextual_recon = model_output
#             contextual_loss = F.mse_loss(contextual_recon, contextual_target)
#             return contextual_loss, torch.tensor(0.0), contextual_loss
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#     print('load model successfully. Start training...')
#     train_start_time = time.time()
#     # 记录损失历史
#     train_losses = []
#     for epoch in range(epochs):
#         print(f'\nEpoch {epoch+1}\n--------------------')
#         epoch_start_time = time.time()
#         epoch_loss = 0
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(train_dataloader):
#             batch_start_time = time.time()
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  
#                 probs = model(batch_contextual_data, batch_contextual_segments)
            
#             # 计算损失
#             loss, temporal_loss, contextual_loss = composite_loss(
#                 probs, batch_temporal_data, batch_contextual_data, model.model_name)
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
#             optimizer.step()
#             optimizer.zero_grad()
            
#             epoch_loss += loss.item()
            
#             batch_end_time = time.time()
#             if batch_idx % 100 == 0:
#                 print(f'loss: {loss.item()}, time cost: {batch_end_time - batch_start_time} s, '
#                       f'[{batch_idx + 1}]/[{len(train_dataloader)}]')
#         avg_epoch_loss = epoch_loss / len(train_dataloader)
#         train_losses.append(avg_epoch_loss)
#         epoch_end_time = time.time()
#         print(f'Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}, '
#               f'Epoch time cost: {epoch_end_time - epoch_start_time} s')
#     train_end_time = time.time()
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label=f'Class Training Loss')
#     plt.title(f'Training Loss Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     print(f'train {model.model_name} Done! Time cost: {train_end_time - train_start_time}')
#     return model

# def test_TrafficScope(data_dir, agg_scales, test_idx,
#                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                       batch_size, model, num_classes, result_path, device,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
#     mse_ls = []
#     test_dataset = TrafficScopeDataset(data_dir, agg_scales, test_idx)
#     if robust_test_name:
#         test_dataset = get_robustness_test_dataset(test_dataset,
#                                                     robust_test_name,
#                                                     rho, kappa, different, alpha, eta, beta, gamma)
#         print(f'generate robust test dataset with {robust_test_name} successfully')
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     model.eval()
#     print('load model successfully. Start testing...')
#     def composite_loss(model_output, temporal_target, contextual_target, model_name):
#         # 与训练时相同的损失函数
#         if model_name == TRAFFIC_SCOPE:
#             temporal_recon, contextual_recon = model_output
#             temporal_loss = F.mse_loss(temporal_recon, temporal_target, reduction='none')
#             contextual_loss = F.mse_loss(contextual_recon, contextual_target, reduction='none')
#             # 对每个样本计算平均损失
#             sample_loss = (temporal_loss.mean(dim=[1,2]) + contextual_loss.mean(dim=[1,2])) / 2
#             return sample_loss
#         elif model_name == TRAFFIC_SCOPE_TEMPORAL:
#             temporal_recon = model_output
#             temporal_loss = F.mse_loss(temporal_recon, temporal_target, reduction='none')
#             return temporal_loss.mean(dim=[1,2])
#         else:  # TRAFFIC_SCOPE_CONTEXTUAL
#             contextual_recon = model_output
#             contextual_loss = F.mse_loss(contextual_recon, contextual_target, reduction='none')
#             return contextual_loss.mean(dim=[1,2])

#     with torch.no_grad():
#         for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
#                         batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(test_dataloader):
#             batch_temporal_data, batch_temporal_valid_len, \
#             batch_contextual_data, batch_contextual_segments, batch_labels = \
#                 batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
#                 batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
#             if model.model_name == TRAFFIC_SCOPE:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len,
#                               batch_contextual_data, batch_contextual_segments)
#             elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
#                 probs = model(batch_temporal_data, batch_temporal_valid_len)
#             else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
#                 probs = model(batch_contextual_data, batch_contextual_segments)
            
#             # 计算每个样本的重构误差
#             test_loss = composite_loss(probs, batch_temporal_data, 
#                                           batch_contextual_data, model.model_name)
            
#             mse_ls.append(test_loss.cpu().numpy()) 
#     return mse_ls

# #方法1：基于分布特性的自适应阈值
# def auto_detect_threshold(loss_list_use):
#     """
#     自动检测loss分布的阈值，不依赖预设的q值
#     """
#     losses = np.array(loss_list_use)
    
#     # 方法1: 基于分布峰度和偏度
#     skewness = stats.skew(losses)
#     kurtosis = stats.kurtosis(losses)
    
#     # 方法2: 使用混合高斯模型拟合
#     try:
#         print('nice')
#         from sklearn.mixture import GaussianMixture
#         losses_reshaped = losses.reshape(-1, 1)
        
#         # 尝试2-3个成分的GMM
#         best_bic = np.inf
#         best_gmm = None
        
#         for n_components in [9, 10, 11, 12, 13, 14]:
#             gmm = GaussianMixture(n_components=n_components, random_state=42)
#             gmm.fit(losses_reshaped)
#             bic = gmm.bic(losses_reshaped)
            
#             if bic < best_bic:
#                 best_bic = bic
#                 best_gmm = gmm
        
#         # 找到最高均值的组分作为异常分布
#         means = best_gmm.means_.flatten()
#         stds = np.sqrt(best_gmm.covariances_.flatten())
#         weights = best_gmm.weights_
        
#         # 选择权重较小但均值较高的组分
#         anomaly_component = np.argmax(means)
#         threshold = means[anomaly_component] - 2 * stds[anomaly_component]
        
#     except:
#         print('bad')
#         # 方法3: 基于统计的fallback方法
#         Q1 = np.percentile(losses, 25)
#         Q3 = np.percentile(losses, 75)
#         IQR = Q3 - Q1
#         threshold = Q3 + 1.5 * IQR
    
#     return max(threshold, np.min(losses))

# #方法2：基于核密度估计的阈值检测
# # def auto_detect_threshold(loss_list_use):
# #     """
# #     基于核密度估计自动检测阈值
# #     """
# #     losses = np.array(loss_list_use)
    
# #     # 核密度估计
# #     kde = gaussian_kde(losses)
# #     x = np.linspace(np.min(losses), np.max(losses), 1000)
# #     density = kde(x)
    
# #     # 找到密度峰值
# #     peaks, _ = find_peaks(density)
    
# #     if len(peaks) >= 2:
# #         # 多峰分布：选择第一个峰后的谷底作为阈值
# #         valley_idx = peaks[0]
# #         for i in range(peaks[0] + 1, len(density) - 1):
# #             if density[i] < density[i-1] and density[i] < density[i+1]:
# #                 valley_idx = i
# #                 break
# #         threshold = x[valley_idx]
# #     else:
# #         # 单峰分布：使用统计方法
# #         mean_loss = np.mean(losses)
# #         std_loss = np.std(losses)
# #         threshold = mean_loss + 2 * std_loss
    
# #     return threshold

# #方法3：基于变化点检测
# # def auto_detect_threshold(loss_list_use):
# #     """
# #     基于变化点检测的阈值确定
# #     """
# #     losses = np.array(loss_list_use)
    
# #     try:
# #         # 对排序后的loss进行变化点检测
# #         sorted_losses = np.sort(losses)
# #         algo = rpt.Pelt(model="rbf").fit(sorted_losses.reshape(-1, 1))
# #         change_points = algo.predict(pen=10)
        
# #         if len(change_points) > 1:
# #             # 第一个变化点作为正常和异常的分界
# #             threshold_idx = change_points[0]
# #             threshold = sorted_losses[threshold_idx]
# #         else:
# #             # Fallback到百分位数方法
# #             threshold = np.percentile(losses, 95)
            
# #     except:
# #         # Fallback方法
# #         threshold = np.percentile(losses, 95)
    
# #     return threshold

# #方法4：基于极值理论（EVT）
# # def auto_detect_threshold(loss_list_use):
# #     q = 5e-2#1e-2
# #     s = SPOT(q)
# #     s.fit(loss_list_use, loss_list_use)
# #     s.initialize()
# #     results = s.run_simp()
# #     if results['thresholds'][0] > 0:
# #         threshold = results['thresholds'][0]    
# #     else:
# #         threshold = np.sort(s.init_data)[int(0.85 * s.init_data.size)]
# #     return threshold

# def train_test_helper(data_dir, agg_scales, model_name, agg_scale_num, agg_points_num, batch_size,
#                       temporal_seq_len, packet_len, freqs_size, use_temporal, use_contextual, is_train, is_test,
#                       num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, result_path,
#                       robust_test_name,
#                       rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None,
#                       k_fold=None, gpu_id=0):
    
#     os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu_id)
    
#     import psutil
#     import numpy as np
    
#     def check_memory_status():
#         # 检查系统内存
#         virtual_memory = psutil.virtual_memory()
#         print(f"总内存: {virtual_memory.total / (1024**3):.2f} GB")
#         print(f"可用内存: {virtual_memory.available / (1024**3):.2f} GB")
#         print(f"已用内存: {virtual_memory.used / (1024**3):.2f} GB")
#         print(f"内存使用率: {virtual_memory.percent}%")
    
#     # 在创建大数组前调用
#     check_memory_status()
    
#     dataset = TrafficScopeDataset(data_dir, agg_scales)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     mod_ls = []
#     thred_ls = []
#     class_ls = []
#     train_num = 1
#     incre_num = 9#2
    
#     count_number = Counter(dataset.labels)
#     total_per_class = np.array(list(count_number.values())).min()
#     test_ratio = 0.3
#     train_per_class = int(total_per_class * (1 - test_ratio))
#     test_per_class = int(total_per_class * test_ratio)
#     total_num = dataset.temporal_data.shape[0]
#     num_classes = len(count_number)  # 动态获取类别数
    
#     allIndex = np.arange(train_num + incre_num)
#     gtlabel = np.zeros(test_per_class * (train_num + incre_num))
#     for pos in range(train_num + incre_num):
#         i = allIndex[pos]
#         gtlabel[pos * test_per_class:(pos + 1) * test_per_class] = i
    
#     print(f"数据集统计: 总样本数={total_num}, 类别数={num_classes}, 每类最小样本数={total_per_class}")
    
#     # 按类别组织索引
#     class_indices = {}
#     for class_id in range(num_classes):
#         class_indices[class_id] = np.where(dataset.labels == class_id)[0]
#         print(f"类别 {class_id}: {len(class_indices[class_id])} 个样本")
    
#     print('=== Train Initializing ===')
#     # indices = np.arange(total_per_class * (train_num + incre_num))
    
#     for class_id in range(train_num):
#         print(f'训练类别 {class_id} 的自编码器...')

#         class_samples = class_indices[class_id]
        
#         if len(class_samples) > train_per_class:
#             train_indices = np.random.choice(class_samples, train_per_class, replace=False)
#         else:
#             train_indices = class_samples
#         model = train_TrafficScope(data_dir, agg_scales, train_indices, batch_size,
#                                         temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                         use_temporal, use_contextual,
#                                         num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device)#model_path,
#         mod_ls.append(model)
#         class_ls.append(class_id)
        
#         mse_ls = test_TrafficScope(data_dir, agg_scales, train_indices,
#                                       temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                       1, model, num_classes, result_path, device,
#                                       robust_test_name,
#                                       rho, kappa, different, alpha, eta, beta, gamma)
        
#         loss_list_use = np.concatenate(mse_ls)
        
#         threshold = auto_detect_threshold(loss_list_use)
            
#         thred_ls.append(threshold)
            
#         print(f'类别 {class_id} 阈值: {threshold:.6f}, 平均误差: {loss_list_use.mean():.6f}')
        
#     print('*** Update model ***')
#     res_ls = []
#     for ind in range(train_num + incre_num):
#         if ind >= train_num:
#             class_samples = class_indices[ind]
            
#             if len(class_samples) > train_per_class:
#                 train_indices = np.random.choice(class_samples, train_per_class, replace=False)
#             else:
#                 train_indices = class_samples
#             model = train_TrafficScope(data_dir, agg_scales, train_indices, batch_size,
#                                             temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
#                                             use_temporal, use_contextual,
#                                             num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device)#model_path,
#             mod_ls.append(model)
#             class_ls.append(ind)
            
#             mse_ls = test_TrafficScope(data_dir, agg_scales, train_indices,
#                                           temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                           1, model, num_classes, result_path, device,
#                                           robust_test_name,
#                                           rho, kappa, different, alpha, eta, beta, gamma)
            
#             loss_list_test = np.concatenate(mse_ls)
            
#             threshold_test = auto_detect_threshold(loss_list_test)
            
#             thred_ls.append(threshold_test)
            
#         mse_test = []
#         for model in mod_ls:
            
#             class_samples = class_indices[ind]
#             print(class_samples)
#             if len(class_samples) > test_per_class:
#                 test_indices = np.setdiff1d(class_samples, train_indices)
#             else:
#                 test_indices = class_samples
#             mse = test_TrafficScope(data_dir, agg_scales, test_indices,
#                                           temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
#                                           1, model, num_classes, result_path, device,
#                                           robust_test_name,
#                                           rho, kappa, different, alpha, eta, beta, gamma)
#             mse_test.append(mse)
        
#         for per in range(test_per_class):
#             mse_test_array = np.array(mse_test)
#             mse_test_array = mse_test_array.squeeze(-1)  
            
#             mse_test_slice = mse_test_array[:, per]
#             cand_res = mse_test_slice[mse_test_slice < np.array(thred_ls)]

#             if len(cand_res) == 0:
#                 res_ls.append(999)
#             else:
#                 min_loss_res = cand_res.min()
#                 mse_test_list = list(mse_test_slice)
#                 res_ls.append(class_ls[mse_test_list.index(min_loss_res)])
                                
#     for ii in range(train_num + incre_num):
#         if ii >= train_num:
#             rep_npy = np.array(res_ls[test_per_class * ii : test_per_class * (ii + 1)])
#             rep_npy2 = rep_npy.copy()
#             rep_npy[rep_npy2==999] = allIndex[ii]
#             res_ls[test_per_class * ii:test_per_class * (ii + 1)] = list(rep_npy)
            
#     y_pred = np.array(res_ls)
#     y_true = np.array(gtlabel)
#     acc = accuracy_score(y_true, y_pred)
#     pre = precision_score(y_true, y_pred, average='macro')
#     rec = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
#     print('Accuracy:', acc)
#     print('Precision:', pre)
#     print('Recall:', rec)
#     print('F1-Score:', f1)

# if __name__ == '__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument('--data_dir', type=str, default='./result', required=False)
#     args.add_argument('--agg_scales', type=str, default='[0, 1, 2]', required=False)
#     args.add_argument('--model_name', type=str, default='TrafficScope', required=False)
#     args.add_argument('--agg_scale_num', type=int, default=3, required=False)
#     args.add_argument('--agg_points_num', type=int, default=128, required=False)
#     args.add_argument('--batch_size', type=int, default=64, required=False)#32
#     args.add_argument('--temporal_seq_len', type=int, default=64, required=False)
#     args.add_argument('--packet_len', type=int, default=64, required=False)
#     args.add_argument('--freqs_size', type=int, default=128, required=False)
#     args.add_argument('--use_temporal', action='store_true', default=True, required=False)
#     args.add_argument('--use_contextual', action='store_true', default=True, required=False)
#     args.add_argument('--is_train', action='store_true', default=True, required=False)
#     args.add_argument('--is_test', action='store_true', default=True, required=False)
#     args.add_argument('--num_heads', type=int, default=8, required=False)
#     args.add_argument('--num_layers', type=int, default=2, required=False)#2
#     args.add_argument('--num_classes', type=int, default=8, required=False)
#     args.add_argument('--dropout', type=float, default=0.2, required=False)#0.5
#     args.add_argument('--lr', type=float, default=1e-4, required=False)#0.001
#     args.add_argument('--epochs', type=int, default=30, required=False)
#     args.add_argument('--model_path', type=str, default='./model_path/model.pth', required=False)
#     args.add_argument('--result_path', type=str, default='./result/test', required=False)
#     args.add_argument('--k_fold', type=int, default=None, required=False)
#     args.add_argument('--gpu_id', type=int, default=0, required=False)
#     args.add_argument('--robust_test_name', type=str, default=None, required=False)
#     args.add_argument('--rho', type=float, default=None, required=False)
#     args.add_argument('--kappa', type=int, default=None, required=False)
#     args.add_argument('--different', action='store_true', default=None, required=False)
#     args.add_argument('--alpha', type=float, default=None, required=False)
#     args.add_argument('--eta', type=int, default=None, required=False)
#     args.add_argument('--beta', type=float, default=None, required=False)
#     args.add_argument('--gamma', type=float, default=None, required=False)
#     args = args.parse_args()
#     # print(args)

#     train_test_helper(args.data_dir, eval(args.agg_scales), args.model_name,
#                       args.agg_scale_num, args.agg_points_num, args.batch_size,
#                       args.temporal_seq_len, args.packet_len, args.freqs_size,
#                       args.use_temporal, args.use_contextual, args.is_train, args.is_test,
#                       args.num_heads, args.num_layers, args.num_classes,
#                       args.dropout, args.lr, args.epochs, args.model_path, args.result_path,
#                       args.robust_test_name,
#                       args.rho, args.kappa, args.different, args.alpha, args.eta, args.beta, args.gamma,
#                       args.k_fold, args.gpu_id)
#之前为两个loss取平均值
######################################################################################
import pandas as pd
from collections import Counter
from evt import SPOT
import torch.nn.functional as F

from matplotlib import pyplot as plt
from scipy import stats

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import ruptures as rpt

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def inject_label_noise_preserve_counts(labels, rho, seed=42, max_attempts=100):
    """
    在 labels 数组上按概率 rho 注入“交叉替换”噪声，保证每个类别的总样本数不变。
    返回：new_labels (ndarray)，actual_changed_fraction（仅在被挑选样本中的实际更改比例）
    labels: 1D numpy array of ints
    rho: float in [0,1]
    seed: random seed for reproducibility
    """
    import numpy as _np
    labels = _np.array(labels).copy()
    _np.random.seed(seed)

    unique_classes, counts = _np.unique(labels, return_counts=True)
    class_indices = {int(c): _np.where(labels == c)[0] for c in unique_classes}

    selected_idx = []
    # 为每个类挑选 m_c 个样本
    for c in unique_classes:
        c = int(c)
        n_c = len(class_indices[c])
        m_c = int(round(rho * n_c))
        if m_c <= 0:
            continue
        sel = _np.random.choice(class_indices[c], size=m_c, replace=False)
        selected_idx.extend(list(sel))

    selected_idx = _np.array(selected_idx, dtype=int)
    if selected_idx.size == 0:
        return labels, 0.0

    orig_labels = labels[selected_idx].copy()
    permuted = orig_labels.copy()

    # 尝试生成一个“有效的随机置换”，尽量避免所有位置都与原标签相同
    for _ in range(max_attempts):
        _np.random.shuffle(permuted)
        # 至少要有一个位置发生变化才认为置换有效；若所有都相同（极少见）则重试
        if not _np.all(permuted == orig_labels):
            break

    labels[selected_idx] = permuted
    changed = _np.sum(labels[selected_idx] != orig_labels)
    actual_changed_fraction = float(changed) / float(selected_idx.size)

    return labels, actual_changed_fraction


def train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
                        temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                        use_temporal, use_contextual,
                        num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device):

    train_dataset = TrafficScopeDataset(data_dir, agg_scales, train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if use_temporal and use_contextual:
        model = TrafficScope(temporal_seq_len, packet_len,
                              freqs_size, agg_scale_num, agg_points_num,
                              num_heads, num_layers, num_classes, dropout)
    elif use_temporal and not use_contextual:
        
        model = TrafficScopeTemporal(temporal_seq_len, packet_len,
                                      num_heads, num_layers, num_classes, dropout)
    elif not use_temporal and use_contextual:
        model = TrafficScopeContextual(agg_scale_num, agg_points_num, freqs_size,
                                        num_heads, num_layers, num_classes, dropout)
    else:
        print('should specify at least one input type')
        return
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    model.to(device)
    # 使用组合损失函数
    def composite_loss(model_output, temporal_target, contextual_target, model_name):
        if model_name == TRAFFIC_SCOPE:
            temporal_recon, contextual_recon = model_output
            temporal_loss = F.mse_loss(temporal_recon, temporal_target)
            contextual_loss = F.mse_loss(contextual_recon, contextual_target)
            return temporal_loss + contextual_loss, temporal_loss, contextual_loss
        elif model_name == TRAFFIC_SCOPE_TEMPORAL:
            temporal_recon = model_output
            temporal_loss = F.mse_loss(temporal_recon, temporal_target)
            return temporal_loss, temporal_loss, torch.tensor(0.0)
        else:  # TRAFFIC_SCOPE_CONTEXTUAL
            contextual_recon = model_output
            contextual_loss = F.mse_loss(contextual_recon, contextual_target)
            return contextual_loss, torch.tensor(0.0), contextual_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print('load model successfully. Start training...')
    train_start_time = time.time()
    # 记录损失历史
    train_losses = []
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}\n--------------------')
        epoch_start_time = time.time()
        epoch_loss = 0
        for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
                        batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(train_dataloader):
            batch_start_time = time.time()
            batch_temporal_data, batch_temporal_valid_len, \
            batch_contextual_data, batch_contextual_segments, batch_labels = \
                batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
                batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
            if model.model_name == TRAFFIC_SCOPE:
                probs = model(batch_temporal_data, batch_temporal_valid_len,
                              batch_contextual_data, batch_contextual_segments)
            elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
                probs = model(batch_temporal_data, batch_temporal_valid_len)
            else:  
                probs = model(batch_contextual_data, batch_contextual_segments)
            
            # 计算损失
            loss, temporal_loss, contextual_loss = composite_loss(
                probs, batch_temporal_data, batch_contextual_data, model.model_name)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            batch_end_time = time.time()
            if batch_idx % 100 == 0:
                print(f'loss: {loss.item()}, time cost: {batch_end_time - batch_start_time} s, '
                      f'[{batch_idx + 1}]/[{len(train_dataloader)}]')
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        epoch_end_time = time.time()
        print(f'Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}, '
              f'Epoch time cost: {epoch_end_time - epoch_start_time} s')
    train_end_time = time.time()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label=f'Class Training Loss')
    plt.title(f'Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    print(f'train {model.model_name} Done! Time cost: {train_end_time - train_start_time}')
    return model

def test_TrafficScope(data_dir, agg_scales, test_idx,
                      temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                      batch_size, model, num_classes, result_path, device,
                      robust_test_name,
                      rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
    temporal_mse_ls = []
    contextual_mse_ls = []
    test_dataset = TrafficScopeDataset(data_dir, agg_scales, test_idx)
    if robust_test_name:
        test_dataset = get_robustness_test_dataset(test_dataset,
                                                    robust_test_name,
                                                    rho, kappa, different, alpha, eta, beta, gamma)
        print(f'generate robust test dataset with {robust_test_name} successfully')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    print('load model successfully. Start testing...')
    def composite_loss(model_output, temporal_target, contextual_target, model_name):
        # 与训练时相同的损失函数
        if model_name == TRAFFIC_SCOPE:
            temporal_recon, contextual_recon = model_output
            temporal_loss = F.mse_loss(temporal_recon, temporal_target, reduction='none')
            contextual_loss = F.mse_loss(contextual_recon, contextual_target, reduction='none')
            # 对每个样本计算平均损失
            # sample_loss = (temporal_loss.mean(dim=[1,2]) + contextual_loss.mean(dim=[1,2])) / 2
            # return sample_loss
            return temporal_loss.mean(dim=[1,2]), contextual_loss.mean(dim=[1,2])
        elif model_name == TRAFFIC_SCOPE_TEMPORAL:
            temporal_recon = model_output
            temporal_loss = F.mse_loss(temporal_recon, temporal_target, reduction='none')
            mean_temporal_loss = temporal_loss.mean(dim=[1,2])
            return mean_temporal_loss, torch.zeros_like(mean_temporal_loss)
        else:  # TRAFFIC_SCOPE_CONTEXTUAL
            contextual_recon = model_output
            contextual_loss = F.mse_loss(contextual_recon, contextual_target, reduction='none')
            mean_contextual_loss = contextual_loss.mean(dim=[1,2])
            return torch.zeros_like(mean_contextual_loss), mean_contextual_loss

    with torch.no_grad():
        for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
                        batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(test_dataloader):
            batch_temporal_data, batch_temporal_valid_len, \
            batch_contextual_data, batch_contextual_segments, batch_labels = \
                batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
                batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
            if model.model_name == TRAFFIC_SCOPE:
                probs = model(batch_temporal_data, batch_temporal_valid_len,
                              batch_contextual_data, batch_contextual_segments)
            elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
                probs = model(batch_temporal_data, batch_temporal_valid_len)
            else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
                probs = model(batch_contextual_data, batch_contextual_segments)
            
            # 计算每个样本的重构误差
            temporal_loss, contextual_loss = composite_loss(probs, batch_temporal_data, 
                                          batch_contextual_data, model.model_name)
            
            temporal_mse_ls.append(temporal_loss.cpu().numpy()) 
            contextual_mse_ls.append(contextual_loss.cpu().numpy()) 
    return temporal_mse_ls, contextual_mse_ls  #mse_ls

#方法1：基于分布特性的自适应阈值
def auto_detect_threshold(loss_list_use):
    """
    自动检测loss分布的阈值，不依赖预设的q值
    """
    losses = np.array(loss_list_use)
    
    # 方法1: 基于分布峰度和偏度
    skewness = stats.skew(losses)
    kurtosis = stats.kurtosis(losses)
    
    # 方法2: 使用混合高斯模型拟合
    try:
        print('nice')
        from sklearn.mixture import GaussianMixture
        losses_reshaped = losses.reshape(-1, 1)
        
        # 尝试2-3个成分的GMM
        best_bic = np.inf
        best_gmm = None
        
        for n_components in [10, 11, 12, 13, 14, 15]:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(losses_reshaped)
            bic = gmm.bic(losses_reshaped)
            
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        
        # 找到最高均值的组分作为异常分布
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_
        
        # 选择权重较小但均值较高的组分
        anomaly_component = np.argmax(means)
        threshold = means[anomaly_component] - 2 * stds[anomaly_component]
        
    except:
        print('bad')
        # 方法3: 基于统计的fallback方法
        Q1 = np.percentile(losses, 25)
        Q3 = np.percentile(losses, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5 * IQR
    
    return max(threshold, np.min(losses))

# 方法2：基于核密度估计的阈值检测
# def auto_detect_threshold(loss_list_use):
#     """
#     基于核密度估计自动检测阈值
#     """
#     losses = np.array(loss_list_use)
    
#     # 核密度估计
#     kde = gaussian_kde(losses)
#     x = np.linspace(np.min(losses), np.max(losses), 1000)
#     density = kde(x)
    
#     # 找到密度峰值
#     peaks, _ = find_peaks(density)
    
#     if len(peaks) >= 2:
#         # 多峰分布：选择第一个峰后的谷底作为阈值
#         valley_idx = peaks[0]
#         for i in range(peaks[0] + 1, len(density) - 1):
#             if density[i] < density[i-1] and density[i] < density[i+1]:
#                 valley_idx = i
#                 break
#         threshold = x[valley_idx]
#     else:
#         # 单峰分布：使用统计方法
#         mean_loss = np.mean(losses)
#         std_loss = np.std(losses)
#         threshold = mean_loss + 2 * std_loss
    
#     return threshold

#方法3：基于变化点检测
# def auto_detect_threshold(loss_list_use):
#     losses = np.array(loss_list_use)
#     sorted_losses = np.sort(losses)
    
#     # 增加惩罚参数，降低敏感性
#     algo = rpt.Pelt(model="rbf").fit(losses.reshape(-1, 1))
#     change_points = algo.predict(pen=50)  # 增加惩罚参数
    
#     if len(change_points) > 1:
#         # 选择更合理的变化点（比如中间位置）
#         print('********************************')
#         print('length:',len(change_points))
#         print('********************************')
#         threshold_idx = change_points[len(change_points)//2]
#         threshold = sorted_losses[threshold_idx]
#     else:
#         # 回退到更保守的方法
#         threshold = np.percentile(losses, 85)  # 降低百分位数
    
#     return threshold

# 方法4：基于极值理论（EVT）
# def auto_detect_threshold(loss_list_use):
#     q = 5e-2
#     s = SPOT(q)
#     s.fit(loss_list_use, loss_list_use)
#     s.initialize()
#     results = s.run_simp()
#     if results['thresholds'][0] > 0:
#         threshold = results['thresholds'][0]    
#     else:
#         threshold = np.sort(s.init_data)[int(0.85 * s.init_data.size)]
#     return threshold

def train_test_helper(data_dir, agg_scales, model_name, agg_scale_num, agg_points_num, batch_size,
                      temporal_seq_len, packet_len, freqs_size, use_temporal, use_contextual, is_train, is_test,
                      num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, result_path,
                      robust_test_name,
                      rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None,
                      k_fold=None, gpu_id=0):
    np.random.seed(42)
    os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu_id)
    
    import psutil
    
    def check_memory_status():
        # 检查系统内存
        virtual_memory = psutil.virtual_memory()
        print(f"总内存: {virtual_memory.total / (1024**3):.2f} GB")
        print(f"可用内存: {virtual_memory.available / (1024**3):.2f} GB")
        print(f"已用内存: {virtual_memory.used / (1024**3):.2f} GB")
        print(f"内存使用率: {virtual_memory.percent}%")
    
    # 在创建大数组前调用
    check_memory_status()
    
    dataset = TrafficScopeDataset(data_dir, agg_scales)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    ##########################################################################################
    ##erro probility of labels 
    # rho = 0.1
    # if rho is not None and rho > 0.0:
    #     print(f'Injecting label noise with rho={rho} ... (preserving per-class counts)')
    #     new_labels, actual_changed_fraction = inject_label_noise_preserve_counts(dataset.labels, rho, seed=42)
    #     # 将 dataset 中的标签就地替换（注意：TrafficScopeDataset 需要使用 self.labels 属性）
    #     dataset.labels = new_labels
    #     # 如果你的 dataset 还有别的缓存（例如 dataset.y 或 labels 列表），也请同步替换
    #     print(f'Label noise injected. Among selected samples actual changed fraction: {actual_changed_fraction:.4f}')
    ########################################################################################
    
    
    
    
    mod_ls = []
    temporal_thred_ls = []
    contextual_thred_ls = []
    class_ls = []
    train_num = 1
    incre_num = 9#8#5#########################################################
    
    count_number = Counter(dataset.labels)
    total_per_class = np.array(list(count_number.values())).min()
    test_ratio = 0.3
    train_per_class = int(total_per_class * (1 - test_ratio))
    test_per_class = int(total_per_class * test_ratio)
    total_num = dataset.temporal_data.shape[0]
    num_classes = len(count_number)
    
    allIndex = np.arange(train_num + incre_num)
    gtlabel = np.zeros(test_per_class * (train_num + incre_num))
    for pos in range(train_num + incre_num):
        i = allIndex[pos]
        gtlabel[pos * test_per_class:(pos + 1) * test_per_class] = i
    
    print(f"数据集统计: 总样本数={total_num}, 类别数={num_classes}, 每类最小样本数={total_per_class}")
    
    # 按类别组织索引
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(dataset.labels == class_id)[0]
        print(f"类别 {class_id}: {len(class_indices[class_id])} 个样本")
    
    print('=== Train Initializing ===')
    # indices = np.arange(total_per_class * (train_num + incre_num))
    
    for class_id in range(train_num):
        print(f'训练类别 {class_id} 的自编码器...')

        class_samples = class_indices[class_id]
        
        if len(class_samples) > train_per_class:
            train_indices = np.random.choice(class_samples, train_per_class, replace=False)
        else:
            train_indices = class_samples
        
        model = train_TrafficScope(data_dir, agg_scales, train_indices, batch_size,
                                        temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                                        use_temporal, use_contextual,
                                        num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device)#model_path,
        mod_ls.append(model)
        class_ls.append(class_id)
        
        temporal_mse_ls, contextual_mse_ls = test_TrafficScope(data_dir, agg_scales, train_indices,
                                      temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                                      1, model, num_classes, result_path, device,
                                      robust_test_name,
                                      rho, kappa, different, alpha, eta, beta, gamma)
        
        temporal_loss_list_use = np.concatenate(temporal_mse_ls)
        contextual_loss_list_use = np.concatenate(contextual_mse_ls)
        
        temporal_threshold = auto_detect_threshold(temporal_loss_list_use)
        contextual_threshold = auto_detect_threshold(contextual_loss_list_use)
            
        temporal_thred_ls.append(temporal_threshold)
        contextual_thred_ls.append(contextual_threshold)
            
        print(f'类别 {class_id} 时域阈值: {temporal_threshold:.6f}, 频域阈值：{contextual_threshold:.6f}, 时域平均误差: {temporal_loss_list_use.mean():.6f}, 频域平均误差：{contextual_loss_list_use.mean():.6f}')
        
    print('*** Update model ***')
    res_ls = []
    for ind in range(train_num + incre_num):
        if ind >= train_num:
            class_samples = class_indices[ind]
            
            if len(class_samples) > train_per_class:
                train_indices = np.random.choice(class_samples, train_per_class, replace=False)
            else:
                train_indices = class_samples
                
            model = train_TrafficScope(data_dir, agg_scales, train_indices, batch_size,
                                            temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                                            use_temporal, use_contextual,
                                            num_heads, num_layers, num_classes, dropout, learning_rate, epochs, device)#model_path,
            mod_ls.append(model)
            class_ls.append(ind)
            
            temporal_mse_ls, contextual_mse_ls = test_TrafficScope(data_dir, agg_scales, train_indices,
                                          temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                                          1, model, num_classes, result_path, device,
                                          robust_test_name,
                                          rho, kappa, different, alpha, eta, beta, gamma)
            
            temporal_loss_list_use = np.concatenate(temporal_mse_ls)
            contextual_loss_list_use = np.concatenate(contextual_mse_ls)
            
            temporal_threshold = auto_detect_threshold(temporal_loss_list_use)
            contextual_threshold = auto_detect_threshold(contextual_loss_list_use)
                
            temporal_thred_ls.append(temporal_threshold)
            contextual_thred_ls.append(contextual_threshold)
            
        temporal_mse_test = []
        contextual_mse_test = []
        for model in mod_ls:
            
            class_samples = class_indices[ind]
            print(class_samples)
            if len(class_samples) > test_per_class:
                test_indices = np.setdiff1d(class_samples, train_indices)
            else:
                test_indices = class_samples

            temporal_mse, contextual_mse = test_TrafficScope(data_dir, agg_scales, test_indices,
                                          temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                                          1, model, num_classes, result_path, device,
                                          robust_test_name,
                                          rho, kappa, different, alpha, eta, beta, gamma)
            temporal_mse_test.append(temporal_mse)
            contextual_mse_test.append(contextual_mse)
        
        for per in range(test_per_class):
            temporal_mse_test_array = np.array(temporal_mse_test).squeeze(-1)
            contextual_mse_test_array = np.array(contextual_mse_test).squeeze(-1)
            
            temporal_mse_slice = temporal_mse_test_array[:, per]
            contextual_mse_slice = contextual_mse_test_array[:, per]
            
            if use_temporal and use_contextual:
                combined_scores = (temporal_mse_slice + contextual_mse_slice)/2
            elif use_temporal and not use_contextual:
                combined_scores = temporal_mse_slice
                
            elif not use_temporal and use_contextual:
                combined_scores = contextual_mse_slice
            else:
                print('should specify at least one input type')
                return
            
            # 找到时域和频域误差最小的索引
            temporal_min_idx = np.argmin(temporal_mse_slice)
            contextual_min_idx = np.argmin(contextual_mse_slice)
            
            if use_temporal and use_contextual:
                cand_mask = (temporal_mse_slice < np.array(temporal_thred_ls)) & (contextual_mse_slice < np.array(contextual_thred_ls)) & (temporal_min_idx == contextual_min_idx)
                cand_res = np.where(cand_mask)[0]
            elif use_temporal and not use_contextual:
                cand_mask = temporal_mse_slice < np.array(temporal_thred_ls)
                cand_res = np.where(cand_mask)[0]
                
            elif not use_temporal and use_contextual:
                cand_mask = contextual_mse_slice < np.array(contextual_thred_ls)
                cand_res = np.where(cand_mask)[0]
            else:
                print('should specify at least one input type')
                return
            
            if len(cand_res) == 0:
                res_ls.append(999)
            else:
                min_loss_idx = cand_res[np.argmin(combined_scores[cand_res])]
                res_ls.append(class_ls[min_loss_idx])
                                
    for ii in range(train_num + incre_num):
        # if ii >= train_num:
        rep_npy = np.array(res_ls[test_per_class * ii : test_per_class * (ii + 1)])
        rep_npy2 = rep_npy.copy()
        rep_npy[rep_npy2==999] = allIndex[ii]
        res_ls[test_per_class * ii:test_per_class * (ii + 1)] = list(rep_npy)
            
    y_pred = np.array(res_ls)
    y_true = np.array(gtlabel)

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print('Accuracy:', acc)
    print('Precision:', pre)
    print('Recall:', rec)
    print('F1-Score:', f1)
    print('AMI:', ami)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default='./multi_result/IDS2017-2018_result', required=False)
    args.add_argument('--agg_scales', type=str, default='[0, 1, 2]', required=False)
    args.add_argument('--model_name', type=str, default='TrafficScope', required=False)
    args.add_argument('--agg_scale_num', type=int, default=3, required=False)
    args.add_argument('--agg_points_num', type=int, default=128, required=False)
    args.add_argument('--batch_size', type=int, default=64, required=False)#32
    args.add_argument('--temporal_seq_len', type=int, default=64, required=False)
    args.add_argument('--packet_len', type=int, default=64, required=False)
    args.add_argument('--freqs_size', type=int, default=128, required=False)
    args.add_argument('--use_temporal', action='store_true', default=True, required=False)
    args.add_argument('--use_contextual', action='store_true', default=True, required=False)
    args.add_argument('--is_train', action='store_true', default=True, required=False)
    args.add_argument('--is_test', action='store_true', default=True, required=False)
    args.add_argument('--num_heads', type=int, default=8, required=False)
    args.add_argument('--num_layers', type=int, default=2, required=False)
    args.add_argument('--num_classes', type=int, default=10, required=False)#8
    args.add_argument('--dropout', type=float, default=0.2, required=False)#0.5
    args.add_argument('--lr', type=float, default=1e-3, required=False)#0.001
    args.add_argument('--epochs', type=int, default=30, required=False)
    args.add_argument('--model_path', type=str, default='./model_path/model.pth', required=False)
    args.add_argument('--result_path', type=str, default='./result/test', required=False)
    args.add_argument('--k_fold', type=int, default=None, required=False)
    args.add_argument('--gpu_id', type=int, default=0, required=False)
    args.add_argument('--robust_test_name', type=str, default='reorder', required=False)#None
    args.add_argument('--rho', type=float, default=None, required=False)
    args.add_argument('--kappa', type=int, default=None, required=False)
    args.add_argument('--different', action='store_true', default=None, required=False)
    args.add_argument('--alpha', type=float, default=None, required=False)
    args.add_argument('--eta', type=int, default=None, required=False)
    args.add_argument('--beta', type=float, default=None, required=False)
    args.add_argument('--gamma', type=float, default=0.5, required=False)#None
    args = args.parse_args()
    # print(args)

    train_test_helper(args.data_dir, eval(args.agg_scales), args.model_name,
                      args.agg_scale_num, args.agg_points_num, args.batch_size,
                      args.temporal_seq_len, args.packet_len, args.freqs_size,
                      args.use_temporal, args.use_contextual, args.is_train, args.is_test,
                      args.num_heads, args.num_layers, args.num_classes,
                      args.dropout, args.lr, args.epochs, args.model_path, args.result_path,
                      args.robust_test_name,
                      args.rho, args.kappa, args.different, args.alpha, args.eta, args.beta, args.gamma,
                      args.k_fold, args.gpu_id)