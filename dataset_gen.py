"""
IDS2017-2018 -> 生成 temporal.npy, mask.npy, contextual.npy, labels.npy, label_map.json
- 每个 flow 取前 64 个包，每包取前 64 字节
- Benign 与每个攻击类别样本量相同（每类上限由参数控制）
- 依赖: Wireshark 工具 (mergecap, tshark, editcap), SplitCap.exe (需在当前目录或PATH)
- 用法: python gen_dataset.py --input_dir ./IDS2017-2018 --out_dir ./out --max_per_class 2000
"""
# import argparse
# import os
# import sys
# import shutil
# import subprocess
# import time
# import json
# import binascii
# import random

# import numpy as np
# import pandas as pd
# import pywt
# import joblib
# import matplotlib
# matplotlib.use('Agg')  # 不显示窗口
# import matplotlib.pyplot as plt

# # -----------------------
# # 参数与环境工具查找
# # -----------------------
# def find_wireshark_tools():
#     wireshark_root = os.environ.get('WIRESHARK', None)
#     candidates = []
#     if wireshark_root:
#         candidates.append(wireshark_root)
#     candidates += [
#         'C:\\Program Files\\Wireshark',
#         'C:\\Program Files (x86)\\Wireshark',
#         '/usr/bin',
#         '/usr/local/bin'
#     ]
#     tools = {}
#     for c in candidates:
#         mc = os.path.join(c, 'mergecap.exe' if os.name == 'nt' else 'mergecap')
#         tc = os.path.join(c, 'tshark.exe' if os.name == 'nt' else 'tshark')
#         ec = os.path.join(c, 'editcap.exe' if os.name == 'nt' else 'editcap')
#         if os.path.exists(mc) and 'mergecap' not in tools:
#             tools['mergecap'] = mc
#         if os.path.exists(tc) and 'tshark' not in tools:
#             tools['tshark'] = tc
#         if os.path.exists(ec) and 'editcap' not in tools:
#             tools['editcap'] = ec
#     # fallback to names (assume in PATH)
#     tools.setdefault('mergecap', 'mergecap')
#     tools.setdefault('tshark', 'tshark')
#     tools.setdefault('editcap', 'editcap')
#     return tools

# TOOLS = find_wireshark_tools()
# # SPLITCAP = os.path.join(os.getcwd(), "SplitCap.exe") if os.name == 'nt' else os.path.join(os.getcwd(), "SplitCap")  # assume available
# SPLITCAP = "SplitCap.exe"
# # -----------------------
# # 辅助 I/O
# # -----------------------
# def list_pcaps_in_dir(d):
#     exts = ('.pcap', '.pcapng')
#     files = []
#     for root, _, fnames in os.walk(d):
#         for f in fnames:
#             if f.lower().endswith(exts):
#                 files.append(os.path.join(root, f))
#     files.sort()
#     return files

# def ensure_dir(d):
#     if not os.path.exists(d):
#         os.makedirs(d)

# # -----------------------
# # 基于你原始代码的 session pcap -> matrix 解析（保持逻辑）
# # -----------------------
# def parse_session_pcap_to_matrix(session_pcap_path, session_len=64, packet_len=64, packet_offset=14):
#     """
#     读取一个 session pcap 文件（raw bytes），返回 (session_matrix, padding_mask)
#     如果 session 太短 (<3 packets)，返回 (None, None)
#     """
#     with open(session_pcap_path, 'rb') as f:
#         content = f.read()
#     hexc = binascii.hexlify(content)

#     if len(hexc) < 48:
#         return None, None

#     # 判断小端顺序
#     if hexc[:8] == b'd4c3b2a1':
#         little_endian = True
#     else:
#         little_endian = False

#     # 全局头 24 bytes -> 48 hex chars
#     hexc = hexc[48:]

#     packets_dec = []
#     while len(hexc) > 0 and len(packets_dec) < session_len:
#         if len(hexc) < 24:
#             break
#         frame_len = hexc[16:24]
#         if little_endian:
#             frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])
#         try:
#             frame_len = int(frame_len, 16)
#         except Exception:
#             break

#         # remove current packet header (16 bytes -> 32 hex chars)
#         hexc = hexc[32:]
#         # get frame bytes, but only take packet_len bytes (with offset)
#         frame_hex = hexc[packet_offset * 2: min(packet_len * 2, frame_len * 2)]
#         frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)] if len(frame_hex) >= 2 else []
#         packets_dec.append(frame_dec)

#         # advance by whole frame_len (in hex length)
#         hexc = hexc[frame_len * 2:]

#     if len(packets_dec) < 3:
#         return None, None

#     # pad into matrix
#     packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int32)
#     session_matrix = np.ones((session_len, packet_len), dtype=np.int16) * -1
#     row_idx = min(packets_dec_matrix.shape[0], session_len)
#     col_idx = min(packets_dec_matrix.shape[1], packet_len)
#     session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

#     # set irrelevant features to -1 (indices adapted by packet_offset)
#     common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
#     tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
#     udp_irr_fea_idx = [40, 41]
#     def sub(idx): return idx - packet_offset
#     try:
#         session_matrix[:, [sub(i) for i in common_irr_fea_idx]] = -1
#     except Exception:
#         pass
#     # set tcp/udp-specific - using protocol field position 23 - packet_offset
#     proto_col = 23 - packet_offset
#     if 0 <= proto_col < session_matrix.shape[1]:
#         # TCP
#         for idx in tcp_irr_fea_idx:
#             c = idx - packet_offset
#             if 0 <= c < session_matrix.shape[1]:
#                 session_matrix[session_matrix[:, proto_col] == 6, c] = -1
#         # UDP
#         for idx in udp_irr_fea_idx:
#             c = idx - packet_offset
#             if 0 <= c < session_matrix.shape[1]:
#                 session_matrix[session_matrix[:, proto_col] == 17, c] = -1

#     return session_matrix.astype(np.int16), (session_matrix == -1).astype(np.uint8)

# # -----------------------
# # Wavelet / contextual 特征（沿用 paper 中的思路）
# # -----------------------
# def wavelet_transform(seq, wave_name='cgau8', agg_points_num=128):
#     """
#     seq: 1D 数组长度 = agg_points_num（或小于，但会被 pad）
#     返回 normalized spectrogram: shape (freqs, t) == (agg_points_num, agg_points_num) 但我们会截取合适部分
#     """
#     seq = np.array(seq, dtype=float)
#     if seq.size < 1:
#         seq = np.zeros(agg_points_num)
#     # pad/truncate to agg_points_num
#     if seq.shape[0] < agg_points_num:
#         seq = np.pad(seq, (0, agg_points_num - seq.shape[0]), 'constant')
#     else:
#         seq = seq[:agg_points_num]

#     scales = np.arange(1, agg_points_num + 1)
#     try:
#         fc = pywt.central_frequency(wave_name)
#     except Exception:
#         fc = pywt.central_frequency('cgau8')
#     scales = 2 * fc * agg_points_num / scales
#     try:
#         cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)
#     except Exception:
#         cwtmatr, freqs = pywt.cwt(seq, scales, 'cgau8')
#     spectrogram = np.log2((np.abs(cwtmatr)) ** 2 + 1)
#     spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-12)
#     return spectrogram  # shape (freqs, t) where freqs == agg_points_num

# # -----------------------
# # 生成某个 class 的 temporal/mask/contextual（内存中返回）
# # -----------------------

# def gen_sessions_with_tshark(filtered_pcap, sessions_dir, session_len=64, packet_len=64, packet_offset=14):
#     """
#     使用 tshark 替代 SplitCap 进行会话分割
#     """
#     tshark = TOOLS['tshark']
    
#     # 获取所有会话的五元组信息
#     cmd = [tshark, '-r', filtered_pcap, '-Y', 'tcp or udp', '-T', 'fields', 
#            '-e', 'frame.time_epoch', '-e', 'ip.src', '-e', 'ip.dst', 
#            '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'udp.srcport', 
#            '-e', 'udp.dstport', '-e', 'ipv6.src', '-e', 'ipv6.dst']
    
#     try:
#         result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         lines = result.stdout.strip().split('\n')
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark session analysis failed: {e}")

#     # 解析会话信息
#     sessions = {}
#     for line in lines:
#         if not line.strip():
#             continue
#         parts = line.split('\t')
#         if len(parts) < 6:
#             continue
            
#         timestamp, src_ip, dst_ip, tcp_sport, tcp_dport, udp_sport, udp_dport = parts[:7]
        
#         # 确定协议和端口
#         if tcp_sport and tcp_dport:
#             sport, dport, proto = tcp_sport, tcp_dport, 'tcp'
#         elif udp_sport and udp_dport:
#             sport, dport, proto = udp_sport, udp_dport, 'udp'
#         else:
#             continue
            
#         # 构建会话键
#         session_key = f"{src_ip}_{sport}_{dst_ip}_{dport}_{proto}"
        
#         if session_key not in sessions:
#             sessions[session_key] = {
#                 'src_ip': src_ip, 'dst_ip': dst_ip,
#                 'sport': sport, 'dport': dport, 'proto': proto
#             }

#     # 为每个会话提取数据包
#     temporal_list = []
#     mask_list = []
#     used_session_paths = []
    
#     for i, (session_key, session_info) in enumerate(sessions.items()):
#         session_file = os.path.join(sessions_dir, f"session_{i}.pcap")
        
#         # 构建显示过滤器
#         if session_info['proto'] == 'tcp':
#             display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and tcp.port=={session_info['sport']} and tcp.port=={session_info['dport']}"
#         else:
#             display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and udp.port=={session_info['sport']} and udp.port=={session_info['dport']}"
        
#         cmd = [tshark, '-r', filtered_pcap, '-Y', display_filter, '-w', session_file]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
#             # 解析会话文件
#             sm, mask = parse_session_pcap_to_matrix(session_file, session_len, packet_len, packet_offset)
#             if sm is not None:
#                 temporal_list.append(sm.astype(np.int16))
#                 mask_list.append(mask.astype(np.uint8))
#                 used_session_paths.append(session_file)
                
#         except subprocess.CalledProcessError:
#             continue

#     return temporal_list, mask_list, used_session_paths, filtered_pcap, sessions_dir

# def gen_class_samples_from_merged_pcap(merged_pcap_path, tmp_dir, session_len=64, packet_len=64, packet_offset=14,
#                                       wave_name='cgau8', agg_points_num=128):
#     """
#     优化版本：直接使用 SplitCap，避免管道
#     """
#     ensure_dir(tmp_dir)
#     base = os.path.splitext(os.path.basename(merged_pcap_path))[0]
#     filtered = os.path.join(tmp_dir, f'{base}_filtered.pcap')
#     sessions_dir = os.path.join(tmp_dir, f'{base}_sessions')
    
#     if os.path.exists(sessions_dir):
#         shutil.rmtree(sessions_dir)
#     os.makedirs(sessions_dir, exist_ok=True)

#     # 1) filter protocols with tshark
#     display_filter = "not (arp or dhcp) and (tcp or udp)"
#     tshark = TOOLS['tshark']
#     cmd = [tshark, '-F', 'pcap', '-r', merged_pcap_path, '-w', filtered, '-Y', display_filter]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark filter error: {e.stderr.decode('utf-8', errors='ignore')}")

#     # 2) 直接使用 SplitCap，避免管道
#     splitcap = SPLITCAP
#     cmd = [
#         splitcap,
#         "-r", filtered,  # 输入文件
#         "-o", sessions_dir,       # 输出文件夹
#         "-s", "session"                 # 按会话分割
#     ]
#     try:
#         subprocess.run(cmd, capture_output=True, text=True, timeout=300)
#     except subprocess.CalledProcessError as e:
#         # 如果 SplitCap 失败，尝试使用 tshark 替代方案
#         print(f"SplitCap failed, trying tshark alternative: {e}")
#         return gen_sessions_with_tshark(filtered, sessions_dir, session_len, packet_len, packet_offset)

#     # 3) parse each session pcap into matrix
#     session_pcaps = list_pcaps_in_dir(sessions_dir)
#     temporal_list = []
#     mask_list = []
#     used_session_paths = []
    
#     for sp in session_pcaps:
#         sm, mask = parse_session_pcap_to_matrix(sp, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset)
#         if sm is None:
#             continue
#         temporal_list.append(sm.astype(np.int16))
#         mask_list.append(mask.astype(np.uint8))
#         used_session_paths.append(sp)

#     return temporal_list, mask_list, used_session_paths, filtered, sessions_dir

# # -----------------------
# # 生成 class 的 contextual 特征（只为选中的 session 列表生成 contextual）
# # -----------------------
# def gen_contextual_for_sessions(merged_pcap_path, selected_session_paths, wave_name='cgau8', agg_points=128):
#     """
#     使用 merged pcap 的 metadata，针对 selected_session_paths（list of session pcap）, 生成 contextual (N x 3 x 128 x 128)
#     这里实现和原论文/代码思路一致的简化流程：
#       - 从 merged_pcap_path 用 tshark 抽取 frame.time_epoch, frame.len, ip.src/dst, ipv6.src/dst, tcp/udp ports, tcp.flags.*
#       - 为每个 session 从文件名解析出五元组，找到 session start time（最小 frame.time_epoch）
#       - 对于每个 session，基于 start time 在全 pcap metadata 上按 ms/s/min 三个尺度聚合成 128 点序列 -> wavelet -> spectrogram
#     返回: contextual_data (N, 3, 128, 128)
#     """
#     if len(selected_session_paths) == 0:
#         return np.zeros((0, 3, agg_points, agg_points), dtype=np.float32)

#     # 1) 用 tshark 抽取 csv 元数据
#     tshark = TOOLS['tshark']
#     csv_tmp = merged_pcap_path + '.meta.csv'
#     fields = '-e frame.time_epoch -e frame.len -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst ' \
#              '-e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport ' \
#              '-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.fin'
#     cmd = [tshark, '-T', 'fields'] + fields.split() + ['-r', merged_pcap_path, '-E', 'header=y', '-E', 'separator=,', '-E', 'occurrence=f']
#     try:
#         with open(csv_tmp, 'w', encoding='utf-8') as f:
#             subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, text=True)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark metadata extract error: {e.stderr.decode('utf-8', errors='ignore')}")

#     pcap_metadata = pd.read_csv(csv_tmp)
#     # 补齐 src/dst/ip 与端口列 (与原代码逻辑类似，但更稳健)
#     # 建立 src_ip/dst_ip 和 src_port/dst_port 列
#     pcap_metadata['src_ip'] = pcap_metadata['ip.src'].fillna(pcap_metadata['ipv6.src'])
#     pcap_metadata['dst_ip'] = pcap_metadata['ip.dst'].fillna(pcap_metadata['ipv6.dst'])
#     # ports
#     pcap_metadata['src_port'] = pcap_metadata['tcp.srcport'].fillna(pcap_metadata['udp.srcport'])
#     pcap_metadata['dst_port'] = pcap_metadata['tcp.dstport'].fillna(pcap_metadata['udp.dstport'])
#     # protocol label: if tcp.srcport not null -> TCP else UDP
#     pcap_metadata['protocol'] = np.where(pcap_metadata['tcp.srcport'].notnull(), 'TCP', 'UDP')

#     # helper to compute five_tuple_key same as original
#     def make_five(row):
#         try:
#             a = str(row['src_ip'])
#             b = str(int(float(row['src_port'])))
#             c = str(row['dst_ip'])
#             d = str(int(float(row['dst_port'])))
#             proto = row['protocol']
#             return '_'.join(sorted([a, b, c, d, proto]))
#         except Exception:
#             return ''
#     pcap_metadata['frame.time_epoch'] = pcap_metadata['frame.time_epoch'].astype(float)
#     pcap_metadata['frame.len'] = pd.to_numeric(pcap_metadata['frame.len'], errors='coerce').fillna(0).astype(float)
#     pcap_metadata['five_tuple_key'] = pcap_metadata.apply(make_five, axis=1)

#     # for each selected session, parse five-tuple from filename (SplitCap 格式: ???.protocol_src-dst_port... 需要健壮解析)
#     contextual_list = []
#     for sp in selected_session_paths:
#         fname = os.path.basename(sp)
#         # original code assumed session name contains ".protocol_srcport_dstip_dstport" style
#         # We try to find a five-tuple key by searching pcap_metadata for same start time candidate
#         # Fallback: compute earliest timestamp of entries in session pcap file itself via tshark
#         try:
#             # get earliest frame.time_epoch in this session pcap
#             cmd = [TOOLS['tshark'], '-r', sp, '-T', 'fields', '-e', 'frame.time_epoch', '-c', '1']
#             res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
#             st = res.stdout.strip().splitlines()
#             if len(st) == 0 or st[0] == '':
#                 session_start_time = None
#             else:
#                 session_start_time = float(st[0].strip())
#         except Exception:
#             session_start_time = None

#         # find the closest frame in pcap_metadata
#         if session_start_time is None:
#             # fallback: use min frame.time_epoch across pcap
#             session_start_time = pcap_metadata['frame.time_epoch'].min()

#         # Build time keys for aggregation
#         def build_agg_seq(agg_scale):
#             # convert epoch -> time_key by dividing by agg_scale and int()
#             time_key = (pcap_metadata['frame.time_epoch'] / agg_scale).map(int)
#             center = int(session_start_time / agg_scale)
#             start = center - agg_points // 2 + 1
#             end = center + agg_points // 2
#             sel = pcap_metadata[(time_key >= start) & (time_key <= end)]
#             grouped = sel.groupby(time_key)['frame.len'].sum()
#             agg = np.zeros(agg_points)
#             for i, val in grouped.items():
#                 idx = int(i - start)
#                 if 0 <= idx < agg_points:
#                     agg[idx] = val
#             return agg

#         ms_seq = build_agg_seq(0.001)
#         s_seq = build_agg_seq(1)
#         min_seq = build_agg_seq(60)

#         ms_spec = wavelet_transform(ms_seq, wave_name=wave_name, agg_points_num=agg_points)
#         s_spec = wavelet_transform(s_seq, wave_name=wave_name, agg_points_num=agg_points)
#         min_spec = wavelet_transform(min_seq, wave_name=wave_name, agg_points_num=agg_points)

#         contextual_list.append(np.stack([ms_spec, s_spec, min_spec], axis=0))  # (3,F,T)

#     contextual_arr = np.array(contextual_list, dtype=np.float32)
#     # cleanup csv_tmp to save space
#     try:
#         os.remove(csv_tmp)
#     except Exception:
#         pass
#     return contextual_arr  # shape (N,3,128,128)

# # -----------------------
# # 合并 class 下所有 pcaps（若目录里有多个），返回一个合并后的临时 pcap 路径
# # -----------------------
# def merge_pcaps(pcaps, out_pcap_path):
#     """
#     pcaps: list of pcap file paths
#     out_pcap_path: 输出文件路径
#     """
#     if len(pcaps) == 0:
#         raise ValueError("no input pcaps to merge")
#     if len(pcaps) == 1:
#         shutil.copy2(pcaps[0], out_pcap_path)
#         return out_pcap_path
#     mergecap = TOOLS['mergecap']
#     cmd = [mergecap, '-F', 'pcap', '-w', out_pcap_path] + pcaps
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"mergecap error: {e.stderr.decode('utf-8', errors='ignore')}")
#     return out_pcap_path

# # -----------------------
# # 主逻辑：遍历输入目录构建平衡数据集
# # -----------------------
# def build_balanced_dataset(input_dir, out_dir, max_per_class=2000,
#                            session_len=64, packet_len=64, packet_offset=14,
#                            wave_name='cgau8', agg_points=128, tmp_root='./tmp_gen'):
#     """
#     input_dir 假设包含 'Benign' 文件夹和 'Attack' 文件夹
#     Attack 中可能是多个 pcap 文件或每类的子目录（比如 DDoS, Bot 等）
#     输出: temporal.npy, mask.npy, contextual.npy, labels.npy, label_map.json 放到 out_dir
#     """
#     ensure_dir(out_dir)
#     ensure_dir(tmp_root)
#     # detect classes
#     benign_dir = os.path.join(input_dir, 'Benign')
#     attack_dir = os.path.join(input_dir, 'Attack')

#     class_sources = {}  # class_name -> list of pcap file paths (can be many)
#     if os.path.isdir(benign_dir):
#         class_sources['Benign'] = list_pcaps_in_dir(benign_dir)
#     else:
#         # fallback: search for any pcap named benign*
#         for f in list_pcaps_in_dir(input_dir):
#             if 'benign' in os.path.basename(f).lower():
#                 class_sources.setdefault('Benign', []).append(f)

#     # Attack: either subdirs or pcap files
#     if os.path.isdir(attack_dir):
#         # if attack dir contains subdirs per attack family
#         for entry in os.listdir(attack_dir):
#             p = os.path.join(attack_dir, entry)
#             if os.path.isdir(p):
#                 pcaps = list_pcaps_in_dir(p)
#                 if pcaps:
#                     class_sources[entry] = pcaps
#             elif os.path.isfile(p) and p.lower().endswith(('.pcap', '.pcapng')):
#                 # treat file name (without ext) as class
#                 name = os.path.splitext(os.path.basename(p))[0]
#                 class_sources[name] = [p]
#     else:
#         # fallback: find pcaps under input dir whose filename indicates attack family
#         for f in list_pcaps_in_dir(input_dir):
#             b = os.path.basename(f)
#             if any(tok.lower() in b.lower() for tok in ['ddos','bot','dos','ssh','ftp','portscan','bruteforce','botnet']):
#                 cls = b.split('.')[0]
#                 class_sources.setdefault(cls, []).append(f)

#     if 'Benign' not in class_sources:
#         raise RuntimeError("Cannot find Benign data under input_dir. Make sure there is a 'Benign' directory.")

#     print("Found classes:", list(class_sources.keys()))
#     label_map = {c: i for i, c in enumerate(sorted(class_sources.keys()))}

#     all_temporal = []
#     all_mask = []
#     all_contextual = []
#     all_labels = []

#     # 先准备 Benign 合并后的 pcap 作为基准
#     merged_cache = {}  # class -> merged pcap path
#     for cls, pcaps in class_sources.items():
#         print(f"[{time.strftime('%H:%M:%S')}] Merging class {cls}: {len(pcaps)} pcap(s)")
#         merged_out = os.path.join(tmp_root, f'{cls}_merged.pcap')
#         if os.path.exists(merged_out):
#             os.remove(merged_out)
#         merge_pcaps(pcaps, merged_out)
#         merged_cache[cls] = merged_out

#     # 为每个 class 生成 temporal 数据（全部），然后根据 Benign 做采样
#     class_samples = {}  # cls -> dict with temporal_list, mask_list, session_paths, filtered_pcap, sessions_dir
#     for cls, merged_pcap in merged_cache.items():
#         print(f"[{time.strftime('%H:%M:%S')}] Generate sessions for class {cls} ...")
#         tdir = os.path.join(tmp_root, f'{cls}_proc')
#         if os.path.exists(tdir):
#             shutil.rmtree(tdir)
#         os.makedirs(tdir, exist_ok=True)
#         temporal_list, mask_list, session_paths, filtered, sessions_dir = gen_class_samples_from_merged_pcap(
#             merged_pcap, tdir, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset,
#             wave_name=wave_name, agg_points_num=agg_points)
#         print(f"  -> {cls}: {len(temporal_list)} sessions extracted")
#         class_samples[cls] = {
#             'temporal': temporal_list,
#             'mask': mask_list,
#             'session_paths': session_paths,
#             'merged_filtered': filtered,
#             'sessions_dir': sessions_dir
#         }

#     # 计算每类采样数：每个攻击类跟 Benign 保持一致，且 <= max_per_class
#     benign_count = len(class_samples['Benign']['temporal'])
#     print(f"Benign sessions: {benign_count}")
#     # 对每个 class，采样数 = min(len(class), benign_count, max_per_class)
#     per_class_counts = {}
#     for cls, info in class_samples.items():
#         cnt = min(len(info['temporal']), benign_count, max_per_class)
#         per_class_counts[cls] = cnt
#         print(f"  will sample {cnt} from class {cls} (available {len(info['temporal'])})")

#     # 随机采样并生成 contextual（针对采样的 session 子集）
#     for cls in sorted(class_samples.keys()):
#         info = class_samples[cls]
#         n_sample = per_class_counts[cls]
#         if n_sample == 0:
#             print(f"  skip class {cls} (0 samples)")
#             continue
#         idxs = list(range(len(info['temporal'])))
#         random.shuffle(idxs)
#         sel_idxs = idxs[:n_sample]
#         # collect temporal/mask
#         temporal_sel = [info['temporal'][i] for i in sel_idxs]
#         mask_sel = [info['mask'][i] for i in sel_idxs]
#         session_sel_paths = [info['session_paths'][i] for i in sel_idxs]
#         # build contextual for these sessions (uses merged_filtered pcap)
#         print(f"[{time.strftime('%H:%M:%S')}] Generating contextual for class {cls} ({n_sample} sessions)...")
#         contextual_arr = gen_contextual_for_sessions(info['merged_filtered'], session_sel_paths, wave_name=wave_name, agg_points=agg_points)
#         # sanity shapes
#         if contextual_arr.shape[0] != n_sample:
#             # 如果 contextual 生成数量不一致，尽量 align
#             min_n = min(contextual_arr.shape[0], n_sample)
#             temporal_sel = temporal_sel[:min_n]
#             mask_sel = mask_sel[:min_n]
#             session_sel_paths = session_sel_paths[:min_n]
#             contextual_arr = contextual_arr[:min_n]
#             n_sample = min_n

#         # append to global
#         all_temporal.extend(temporal_sel)
#         all_mask.extend(mask_sel)
#         all_contextual.append(contextual_arr)  # we'll vstack later
#         all_labels.extend([label_map[cls]] * n_sample)

#         print(f"  appended {n_sample} samples of class {cls}")

#     # concat and save
#     print("[*] Concatenating and saving...")
#     temporal_np = np.stack(all_temporal, axis=0).astype(np.int16)  # (N,64,64)
#     mask_np = np.stack(all_mask, axis=0).astype(np.uint8)
#     contextual_np = np.vstack(all_contextual).astype(np.float32) if len(all_contextual) > 0 else np.zeros((0,3,agg_points,agg_points), dtype=np.float32)
#     labels_np = np.array(all_labels, dtype=np.int32)

#     np.save(os.path.join(out_dir, 'temporal.npy'), temporal_np)
#     np.save(os.path.join(out_dir, 'mask.npy'), mask_np)
#     np.save(os.path.join(out_dir, 'contextual.npy'), contextual_np)
#     np.save(os.path.join(out_dir, 'labels.npy'), labels_np)
#     with open(os.path.join(out_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
#         json.dump(label_map, f, indent=2, ensure_ascii=False)

#     print(f"[DONE] Saved: temporal.npy ({temporal_np.shape}), mask.npy ({mask_np.shape}), contextual.npy ({contextual_np.shape}), labels.npy ({labels_np.shape})")
#     return {
#         'temporal_shape': temporal_np.shape,
#         'mask_shape': mask_np.shape,
#         'contextual_shape': contextual_np.shape,
#         'labels_shape': labels_np.shape,
#         'label_map': label_map
#     }

# # -----------------------
# # 主入口
# # -----------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, default='./original_data/CrossNet2021_pcap/ScenarioB', help='IDS2017-2018 数据目录，包含 Benign 和 Attack 子目录')
#     parser.add_argument('--out_dir', type=str, default='./gene_data_all', help='输出目录')
#     parser.add_argument('--max_per_class', type=int, default=100, help='每个类的最大样本数（以控制运行时间）')
#     parser.add_argument('--tmp_root', type=str, default='./tmp_gen', help='临时文件目录')
#     parser.add_argument('--wave', type=str, default='cgau8', help='母 wavelet，默认 cgau8')
#     parser.add_argument('--session_len', type=int, default=64)
#     parser.add_argument('--packet_len', type=int, default=64)
#     parser.add_argument('--packet_offset', type=int, default=14)
#     parser.add_argument('--agg_points', type=int, default=128)
#     args = parser.parse_args()

#     random.seed(42)
#     np.random.seed(42)

#     t0 = time.time()
#     res = build_balanced_dataset(args.input_dir, args.out_dir, max_per_class=args.max_per_class,
#                                  session_len=args.session_len, packet_len=args.packet_len, packet_offset=args.packet_offset,
#                                  wave_name=args.wave, agg_points=args.agg_points, tmp_root=args.tmp_root)
#     t1 = time.time()
#     print("Total time: %.2f s" % (t1 - t0))
#     print(res)


#################################################scenarioA---------scenarioB
# import argparse
# import os
# import sys
# import shutil
# import subprocess
# import time
# import json
# import binascii
# import random

# import numpy as np
# import pandas as pd
# import pywt
# import joblib
# import matplotlib
# matplotlib.use('Agg')  # 不显示窗口
# import matplotlib.pyplot as plt
# import re

# # -----------------------
# # 参数与环境工具查找
# # -----------------------
# def find_wireshark_tools():
#     wireshark_root = os.environ.get('WIRESHARK', None)
#     candidates = []
#     if wireshark_root:
#         candidates.append(wireshark_root)
#     candidates += [
#         'C:\\Program Files\\Wireshark',
#         'C:\\Program Files (x86)\\Wireshark',
#         '/usr/bin',
#         '/usr/local/bin'
#     ]
#     tools = {}
#     for c in candidates:
#         mc = os.path.join(c, 'mergecap.exe' if os.name == 'nt' else 'mergecap')
#         tc = os.path.join(c, 'tshark.exe' if os.name == 'nt' else 'tshark')
#         ec = os.path.join(c, 'editcap.exe' if os.name == 'nt' else 'editcap')
#         if os.path.exists(mc) and 'mergecap' not in tools:
#             tools['mergecap'] = mc
#         if os.path.exists(tc) and 'tshark' not in tools:
#             tools['tshark'] = tc
#         if os.path.exists(ec) and 'editcap' not in tools:
#             tools['editcap'] = ec
#     # fallback to names (assume in PATH)
#     tools.setdefault('mergecap', 'mergecap')
#     tools.setdefault('tshark', 'tshark')
#     tools.setdefault('editcap', 'editcap')
#     return tools

# TOOLS = find_wireshark_tools()
# # SPLITCAP = os.path.join(os.getcwd(), "SplitCap.exe") if os.name == 'nt' else os.path.join(os.getcwd(), "SplitCap")  # assume available
# SPLITCAP = "SplitCap.exe"
# # -----------------------
# # 辅助 I/O
# # -----------------------
# def list_pcaps_in_dir(d):
#     exts = ('.pcap', '.pcapng')
#     files = []
#     for root, _, fnames in os.walk(d):
#         for f in fnames:
#             if f.lower().endswith(exts):
#                 files.append(os.path.join(root, f))
#     files.sort()
#     return files

# def ensure_dir(d):
#     if not os.path.exists(d):
#         os.makedirs(d)

# # -----------------------
# # 基于你原始代码的 session pcap -> matrix 解析（保持逻辑）
# # -----------------------
# def parse_session_pcap_to_matrix(session_pcap_path, session_len=64, packet_len=64, packet_offset=14):
#     """
#     读取一个 session pcap 文件（raw bytes），返回 (session_matrix, padding_mask)
#     如果 session 太短 (<3 packets)，返回 (None, None)
#     """
#     with open(session_pcap_path, 'rb') as f:
#         content = f.read()
#     hexc = binascii.hexlify(content)

#     if len(hexc) < 48:
#         return None, None

#     # 判断小端顺序
#     if hexc[:8] == b'd4c3b2a1':
#         little_endian = True
#     else:
#         little_endian = False

#     # 全局头 24 bytes -> 48 hex chars
#     hexc = hexc[48:]

#     packets_dec = []
#     while len(hexc) > 0 and len(packets_dec) < session_len:
#         if len(hexc) < 24:
#             break
#         frame_len = hexc[16:24]
#         if little_endian:
#             frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])
#         try:
#             frame_len = int(frame_len, 16)
#         except Exception:
#             break

#         # remove current packet header (16 bytes -> 32 hex chars)
#         hexc = hexc[32:]
#         # get frame bytes, but only take packet_len bytes (with offset)
#         frame_hex = hexc[packet_offset * 2: min(packet_len * 2, frame_len * 2)]
#         frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)] if len(frame_hex) >= 2 else []
#         packets_dec.append(frame_dec)

#         # advance by whole frame_len (in hex length)
#         hexc = hexc[frame_len * 2:]

#     if len(packets_dec) < 3:
#         return None, None

#     # pad into matrix
#     packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int32)
#     session_matrix = np.ones((session_len, packet_len), dtype=np.int16) * -1
#     row_idx = min(packets_dec_matrix.shape[0], session_len)
#     col_idx = min(packets_dec_matrix.shape[1], packet_len)
#     session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

#     # set irrelevant features to -1 (indices adapted by packet_offset)
#     common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
#     tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
#     udp_irr_fea_idx = [40, 41]
#     def sub(idx): return idx - packet_offset
#     try:
#         session_matrix[:, [sub(i) for i in common_irr_fea_idx]] = -1
#     except Exception:
#         pass
#     # set tcp/udp-specific - using protocol field position 23 - packet_offset
#     proto_col = 23 - packet_offset
#     if 0 <= proto_col < session_matrix.shape[1]:
#         # TCP
#         for idx in tcp_irr_fea_idx:
#             c = idx - packet_offset
#             if 0 <= c < session_matrix.shape[1]:
#                 session_matrix[session_matrix[:, proto_col] == 6, c] = -1
#         # UDP
#         for idx in udp_irr_fea_idx:
#             c = idx - packet_offset
#             if 0 <= c < session_matrix.shape[1]:
#                 session_matrix[session_matrix[:, proto_col] == 17, c] = -1

#     return session_matrix.astype(np.int16), (session_matrix == -1).astype(np.uint8)

# # -----------------------
# # Wavelet / contextual 特征（沿用 paper 中的思路）
# # -----------------------
# def wavelet_transform(seq, wave_name='cgau8', agg_points_num=128):
#     """
#     seq: 1D 数组长度 = agg_points_num（或小于，但会被 pad）
#     返回 normalized spectrogram: shape (freqs, t) == (agg_points_num, agg_points_num) 但我们会截取合适部分
#     """
#     seq = np.array(seq, dtype=float)
#     if seq.size < 1:
#         seq = np.zeros(agg_points_num)
#     # pad/truncate to agg_points_num
#     if seq.shape[0] < agg_points_num:
#         seq = np.pad(seq, (0, agg_points_num - seq.shape[0]), 'constant')
#     else:
#         seq = seq[:agg_points_num]

#     scales = np.arange(1, agg_points_num + 1)
#     try:
#         fc = pywt.central_frequency(wave_name)
#     except Exception:
#         fc = pywt.central_frequency('cgau8')
#     scales = 2 * fc * agg_points_num / scales
#     try:
#         cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)
#     except Exception:
#         cwtmatr, freqs = pywt.cwt(seq, scales, 'cgau8')
#     spectrogram = np.log2((np.abs(cwtmatr)) ** 2 + 1)
#     spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-12)
#     return spectrogram  # shape (freqs, t) where freqs == agg_points_num

# # -----------------------
# # 生成某个 class 的 temporal/mask/contextual（内存中返回）
# # -----------------------

# def gen_sessions_with_tshark(filtered_pcap, sessions_dir, session_len=64, packet_len=64, packet_offset=14):
#     """
#     使用 tshark 替代 SplitCap 进行会话分割
#     """
#     tshark = TOOLS['tshark']
    
#     # 获取所有会话的五元组信息
#     cmd = [tshark, '-r', filtered_pcap, '-Y', 'tcp or udp', '-T', 'fields', 
#             '-e', 'frame.time_epoch', '-e', 'ip.src', '-e', 'ip.dst', 
#             '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'udp.srcport', 
#             '-e', 'udp.dstport', '-e', 'ipv6.src', '-e', 'ipv6.dst']
    
#     try:
#         result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         lines = result.stdout.strip().split('\n')
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark session analysis failed: {e}")

#     # 解析会话信息
#     sessions = {}
#     for line in lines:
#         if not line.strip():
#             continue
#         parts = line.split('\t')
#         if len(parts) < 6:
#             continue
            
#         timestamp, src_ip, dst_ip, tcp_sport, tcp_dport, udp_sport, udp_dport = parts[:7]
        
#         # 确定协议和端口
#         if tcp_sport and tcp_dport:
#             sport, dport, proto = tcp_sport, tcp_dport, 'tcp'
#         elif udp_sport and udp_dport:
#             sport, dport, proto = udp_sport, udp_dport, 'udp'
#         else:
#             continue
            
#         # 构建会话键
#         session_key = f"{src_ip}_{sport}_{dst_ip}_{dport}_{proto}"
        
#         if session_key not in sessions:
#             sessions[session_key] = {
#                 'src_ip': src_ip, 'dst_ip': dst_ip,
#                 'sport': sport, 'dport': dport, 'proto': proto
#             }

#     # 为每个会话提取数据包
#     temporal_list = []
#     mask_list = []
#     used_session_paths = []
    
#     for i, (session_key, session_info) in enumerate(sessions.items()):
#         session_file = os.path.join(sessions_dir, f"session_{i}.pcap")
        
#         # 构建显示过滤器
#         if session_info['proto'] == 'tcp':
#             display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and tcp.port=={session_info['sport']} and tcp.port=={session_info['dport']}"
#         else:
#             display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and udp.port=={session_info['sport']} and udp.port=={session_info['dport']}"
        
#         cmd = [tshark, '-r', filtered_pcap, '-Y', display_filter, '-w', session_file]
        
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
#             # 解析会话文件
#             sm, mask = parse_session_pcap_to_matrix(session_file, session_len, packet_len, packet_offset)
#             if sm is not None:
#                 temporal_list.append(sm.astype(np.int16))
#                 mask_list.append(mask.astype(np.uint8))
#                 used_session_paths.append(session_file)
                
#         except subprocess.CalledProcessError:
#             continue

#     return temporal_list, mask_list, used_session_paths, filtered_pcap, sessions_dir

# def gen_class_samples_from_merged_pcap(merged_pcap_path, tmp_dir, session_len=64, packet_len=64, packet_offset=14,
#                                       wave_name='cgau8', agg_points_num=128):
#     """
#     优化版本：直接使用 SplitCap，避免管道
#     """
#     ensure_dir(tmp_dir)
#     base = os.path.splitext(os.path.basename(merged_pcap_path))[0]
#     filtered = os.path.join(tmp_dir, f'{base}_filtered.pcap')
#     sessions_dir = os.path.join(tmp_dir, f'{base}_sessions')
    
#     if os.path.exists(sessions_dir):
#         shutil.rmtree(sessions_dir)
#     os.makedirs(sessions_dir, exist_ok=True)

#     # 1) filter protocols with tshark
#     display_filter = "not (arp or dhcp) and (tcp or udp)"
#     tshark = TOOLS['tshark']
#     cmd = [tshark, '-F', 'pcap', '-r', merged_pcap_path, '-w', filtered, '-Y', display_filter]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark filter error: {e.stderr.decode('utf-8', errors='ignore')}")

#     # 2) 直接使用 SplitCap，避免管道
#     splitcap = SPLITCAP
#     cmd = [
#         splitcap,
#         "-r", filtered,  # 输入文件
#         "-o", sessions_dir,       # 输出文件夹
#         "-s", "session"                 # 按会话分割
#     ]
#     try:
#         subprocess.run(cmd, capture_output=True, text=True, timeout=300)
#     except subprocess.CalledProcessError as e:
#         # 如果 SplitCap 失败，尝试使用 tshark 替代方案
#         print(f"SplitCap failed, trying tshark alternative: {e}")
#         return gen_sessions_with_tshark(filtered, sessions_dir, session_len, packet_len, packet_offset)

#     # 3) parse each session pcap into matrix
#     session_pcaps = list_pcaps_in_dir(sessions_dir)
#     temporal_list = []
#     mask_list = []
#     used_session_paths = []
    
#     for sp in session_pcaps:
#         sm, mask = parse_session_pcap_to_matrix(sp, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset)
#         if sm is None:
#             continue
#         temporal_list.append(sm.astype(np.int16))
#         mask_list.append(mask.astype(np.uint8))
#         used_session_paths.append(sp)

#     return temporal_list, mask_list, used_session_paths, filtered, sessions_dir

# # -----------------------
# # 生成 class 的 contextual 特征（只为选中的 session 列表生成 contextual）
# # -----------------------
# def gen_contextual_for_sessions(merged_pcap_path, selected_session_paths, wave_name='cgau8', agg_points=128):
#     """
#     使用 merged pcap 的 metadata，针对 selected_session_paths（list of session pcap）, 生成 contextual (N x 3 x 128 x 128)
#     这里实现和原论文/代码思路一致的简化流程：
#       - 从 merged_pcap_path 用 tshark 抽取 frame.time_epoch, frame.len, ip.src/dst, ipv6.src/dst, tcp/udp ports, tcp.flags.*
#       - 为每个 session 从文件名解析出五元组，找到 session start time（最小 frame.time_epoch）
#       - 对于每个 session，基于 start time 在全 pcap metadata 上按 ms/s/min 三个尺度聚合成 128 点序列 -> wavelet -> spectrogram
#     返回: contextual_data (N, 3, 128, 128)
#     """
#     if len(selected_session_paths) == 0:
#         return np.zeros((0, 3, agg_points, agg_points), dtype=np.float32)

#     # 1) 用 tshark 抽取 csv 元数据
#     tshark = TOOLS['tshark']
#     csv_tmp = merged_pcap_path + '.meta.csv'
#     fields = '-e frame.time_epoch -e frame.len -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst ' \
#               '-e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport ' \
#               '-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.fin'
#     cmd = [tshark, '-T', 'fields'] + fields.split() + ['-r', merged_pcap_path, '-E', 'header=y', '-E', 'separator=,', '-E', 'occurrence=f']
#     try:
#         with open(csv_tmp, 'w', encoding='utf-8') as f:
#             subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, text=True)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"tshark metadata extract error: {e.stderr.decode('utf-8', errors='ignore')}")

#     pcap_metadata = pd.read_csv(csv_tmp)
#     # 补齐 src/dst/ip 与端口列 (与原代码逻辑类似，但更稳健)
#     # 建立 src_ip/dst_ip 和 src_port/dst_port 列
#     pcap_metadata['src_ip'] = pcap_metadata['ip.src'].fillna(pcap_metadata['ipv6.src'])
#     pcap_metadata['dst_ip'] = pcap_metadata['ip.dst'].fillna(pcap_metadata['ipv6.dst'])
#     # ports
#     pcap_metadata['src_port'] = pcap_metadata['tcp.srcport'].fillna(pcap_metadata['udp.srcport'])
#     pcap_metadata['dst_port'] = pcap_metadata['tcp.dstport'].fillna(pcap_metadata['udp.dstport'])
#     # protocol label: if tcp.srcport not null -> TCP else UDP
#     pcap_metadata['protocol'] = np.where(pcap_metadata['tcp.srcport'].notnull(), 'TCP', 'UDP')

#     # helper to compute five_tuple_key same as original
#     def make_five(row):
#         try:
#             a = str(row['src_ip'])
#             b = str(int(float(row['src_port'])))
#             c = str(row['dst_ip'])
#             d = str(int(float(row['dst_port'])))
#             proto = row['protocol']
#             return '_'.join(sorted([a, b, c, d, proto]))
#         except Exception:
#             return ''
#     pcap_metadata['frame.time_epoch'] = pcap_metadata['frame.time_epoch'].astype(float)
#     pcap_metadata['frame.len'] = pd.to_numeric(pcap_metadata['frame.len'], errors='coerce').fillna(0).astype(float)
#     pcap_metadata['five_tuple_key'] = pcap_metadata.apply(make_five, axis=1)

#     # for each selected session, parse five-tuple from filename (SplitCap 格式: ???.protocol_src-dst_port... 需要健壮解析)
#     contextual_list = []
#     for sp in selected_session_paths:
#         fname = os.path.basename(sp)
#         # original code assumed session name contains ".protocol_srcport_dstip_dstport" style
#         # We try to find a five-tuple key by searching pcap_metadata for same start time candidate
#         # Fallback: compute earliest timestamp of entries in session pcap file itself via tshark
#         try:
#             # get earliest frame.time_epoch in this session pcap
#             cmd = [TOOLS['tshark'], '-r', sp, '-T', 'fields', '-e', 'frame.time_epoch', '-c', '1']
#             res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
#             st = res.stdout.strip().splitlines()
#             if len(st) == 0 or st[0] == '':
#                 session_start_time = None
#             else:
#                 session_start_time = float(st[0].strip())
#         except Exception:
#             session_start_time = None

#         # find the closest frame in pcap_metadata
#         if session_start_time is None:
#             # fallback: use min frame.time_epoch across pcap
#             session_start_time = pcap_metadata['frame.time_epoch'].min()

#         # Build time keys for aggregation
#         def build_agg_seq(agg_scale):
#             # convert epoch -> time_key by dividing by agg_scale and int()
#             time_key = (pcap_metadata['frame.time_epoch'] / agg_scale).map(int)
#             center = int(session_start_time / agg_scale)
#             start = center - agg_points // 2 + 1
#             end = center + agg_points // 2
#             sel = pcap_metadata[(time_key >= start) & (time_key <= end)]
#             grouped = sel.groupby(time_key)['frame.len'].sum()
#             agg = np.zeros(agg_points)
#             for i, val in grouped.items():
#                 idx = int(i - start)
#                 if 0 <= idx < agg_points:
#                     agg[idx] = val
#             return agg

#         ms_seq = build_agg_seq(0.001)
#         s_seq = build_agg_seq(1)
#         min_seq = build_agg_seq(60)

#         ms_spec = wavelet_transform(ms_seq, wave_name=wave_name, agg_points_num=agg_points)
#         s_spec = wavelet_transform(s_seq, wave_name=wave_name, agg_points_num=agg_points)
#         min_spec = wavelet_transform(min_seq, wave_name=wave_name, agg_points_num=agg_points)

#         contextual_list.append(np.stack([ms_spec, s_spec, min_spec], axis=0))  # (3,F,T)

#     contextual_arr = np.array(contextual_list, dtype=np.float32)
#     # cleanup csv_tmp to save space
#     try:
#         os.remove(csv_tmp)
#     except Exception:
#         pass
#     return contextual_arr  # shape (N,3,128,128)

# # -----------------------
# # 合并 class 下所有 pcaps（若目录里有多个），返回一个合并后的临时 pcap 路径
# # -----------------------
# def merge_pcaps(pcaps, out_pcap_path):
#     """
#     pcaps: list of pcap file paths
#     out_pcap_path: 输出文件路径
#     """
#     if len(pcaps) == 0:
#         raise ValueError("no input pcaps to merge")
#     if len(pcaps) == 1:
#         shutil.copy2(pcaps[0], out_pcap_path)
#         return out_pcap_path
#     mergecap = TOOLS['mergecap']
#     cmd = [mergecap, '-F', 'pcap', '-w', out_pcap_path] + pcaps
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"mergecap error: {e.stderr.decode('utf-8', errors='ignore')}")
#     return out_pcap_path

# # -----------------------
# # 主逻辑：遍历输入目录构建平衡数据集
# # -----------------------
# def build_balanced_dataset(input_dir, out_dir, max_per_class=2000,
#                             session_len=64, packet_len=64, packet_offset=14,
#                             wave_name='cgau8', agg_points=128, tmp_root='./tmp_gen'):
#     """
#     input_dir 假设包含 'Benign' 文件夹和 'Attack' 文件夹
#     Attack 中可能是多个 pcap 文件或每类的子目录（比如 DDoS, Bot 等）
#     输出: temporal.npy, mask.npy, contextual.npy, labels.npy, label_map.json 放到 out_dir
#     """
#     ensure_dir(out_dir)
#     ensure_dir(tmp_root)
#     # detect classes
#     benign_dir = os.path.join(input_dir, 'Benign')
#     attack_dir = os.path.join(input_dir, 'Attack')

#     class_sources = {}  # class_name -> list of pcap file paths (can be many)
#     if os.path.isdir(benign_dir):
#         class_sources['Benign'] = list_pcaps_in_dir(benign_dir)
#     else:
#         # fallback: search for any pcap named benign*
#         for f in list_pcaps_in_dir(input_dir):
#             if 'benign' in os.path.basename(f).lower():
#                 class_sources.setdefault('Benign', []).append(f)

#     # Attack: either subdirs or pcap files
#     if os.path.isdir(attack_dir):
#         # if attack dir contains subdirs per attack family
#         for entry in os.listdir(attack_dir):
#             p = os.path.join(attack_dir, entry)
#             if os.path.isdir(p):
#                 pcaps = list_pcaps_in_dir(p)
#                 if pcaps:
#                     class_sources[entry] = pcaps
#             elif os.path.isfile(p) and p.lower().endswith(('.pcap', '.pcapng')):
#                 # treat file name (without ext) as class
#                 name = os.path.splitext(os.path.basename(p))[0]
#                 class_sources[name] = [p]
#     else:
#         # fallback: find pcaps under input dir whose filename indicates attack family
#         for f in list_pcaps_in_dir(input_dir):
#             b = os.path.basename(f)
#             if any(tok.lower() in b.lower() for tok in ['ddos','bot','dos','ssh','ftp','portscan','bruteforce','botnet']):
#                 cls = b.split('.')[0]
#                 class_sources.setdefault(cls, []).append(f)

#     if 'Benign' not in class_sources:
#         raise RuntimeError("Cannot find Benign data under input_dir. Make sure there is a 'Benign' directory.")

#     desired_classes = ['Benign', 'aiqiyi', 'baidupan', 'csdn', 'huya',
#                        'microsoft', 'sohu', 'sougou', 'TIM', 'zhihu']

#     print("Found classes (detected):", list(class_sources.keys()))
#     print("Desired classes (will be used to build dataset):", desired_classes)

#     # 使用 desired_classes 构建 label_map，避免 KeyError（确保顺序稳定）
#     label_map = {c: i for i, c in enumerate(sorted(desired_classes))}
#     print("Label map:", label_map)

#     all_temporal = []
#     all_mask = []
#     all_contextual = []
#     all_labels = []

#     # 先准备 Benign 合并后的 pcap 作为基准
#     # merged_cache = {}  # class -> merged pcap path
#     # for cls, pcaps in class_sources.items():
#     #     print(f"[{time.strftime('%H:%M:%S')}] Merging class {cls}: {len(pcaps)} pcap(s)")
#     #     merged_out = os.path.join(tmp_root, f'{cls}_merged.pcap')
#     #     if os.path.exists(merged_out):
#     #         os.remove(merged_out)
#     #     merge_pcaps(pcaps, merged_out)
#     #     merged_cache[cls] = merged_out
    
#     desired_classes = ['Benign', 'aiqiyi', 'baidupan', 'csdn', 'huya',
#                        'microsoft', 'sohu', 'sougou', 'TIM', 'zhihu']
#     desired_per_class = 250  # 每类最终希望获得的样本数

#     scenario_b_dir = os.path.abspath(input_dir)
#     # 尝试在同级目录寻找 ScenarioA
#     parent_dir = os.path.dirname(scenario_b_dir.rstrip(os.sep))
#     scenario_a_dir = os.path.join(parent_dir, 'ScenarioA')
#     if not os.path.isdir(scenario_a_dir):
#         # 尝试直接替换 ScenarioB->ScenarioA（兼容不同路径写法）
#         if 'ScenarioB' in scenario_b_dir:
#             scenario_a_dir = scenario_b_dir.replace('ScenarioB', 'ScenarioA')
#         scenario_a_dir = os.path.abspath(scenario_a_dir)

#     print(f"ScenarioB: {scenario_b_dir}")
#     print(f"ScenarioA (supplement search path): {scenario_a_dir}")

#     def find_pcaps_for_class(clsname, search_dir):
#         """在目录下递归查找与类名匹配的 pcap 文件（更宽松匹配策略），并返回 sorted list。"""
#         found = []
#         if not os.path.isdir(search_dir):
#             return found
#         token = clsname.lower()
#         # 常见别名/拼写修正表（可按需扩展）
#         alias_map = {
#             'aiqiyi': ['iqiyi', 'ai_qiyi', 'ai-qiyi'],
#             'baidupan': ['baidunpan', 'baidu_pan'],
#             'sougou': ['sogou'],
#             'TIM': ['tim'],
#             'zhihu': ['zh', 'zhihu']
#         }
#         aliases = alias_map.get(clsname, []) + [token]

#         for p in list_pcaps_in_dir(search_dir):
#             bn = os.path.basename(p).lower()
#             bn_no_ext = os.path.splitext(bn)[0]
#             # 规则：文件名包含 token/别名 或 文件名前缀等 -> 认为匹配
#             matched = False
#             for a in aliases:
#                 if a in bn or bn_no_ext == a or bn_no_ext.startswith(a + '_') or bn_no_ext.startswith(a + '-'):
#                     matched = True
#                     break
#             # 额外尝试：移除非字母数字后再匹配（容错）
#             if not matched:
#                 bn_simple = re.sub(r'[^a-z0-9]', '', bn_no_ext)
#                 for a in aliases:
#                     if a.replace('_','') in bn_simple:
#                         matched = True
#                         break
#             if matched:
#                 found.append(p)
#         found = sorted(list(dict.fromkeys(found)))
#         if len(found) > 0:
#             print(f"  find_pcaps_for_class('{clsname}') found {len(found)} files (sample 5): {found[:5]}")
#         else:
#             print(f"  find_pcaps_for_class('{clsname}') found NONE under {search_dir}")
#         return found

#     class_samples = {}
#     for cls in desired_classes:
#         print(f"[{time.strftime('%H:%M:%S')}] Preparing merged pcap for class {cls} from ScenarioB...")
#         pcaps_b = find_pcaps_for_class(cls, scenario_b_dir)
#         merged_out = os.path.join(tmp_root, f'{cls}_merged.pcap')
#         # 清理旧产物
#         if os.path.exists(merged_out):
#             try:
#                 os.remove(merged_out)
#             except Exception:
#                 pass

#         if len(pcaps_b) == 0:
#             print(f"  WARNING: no pcap for class {cls} found in ScenarioB ({scenario_b_dir}). Will try ScenarioA later.")
#         else:
#             # 合并 ScenarioB 中找到的 pcaps（如果只有一个则只是拷贝）
#             try:
#                 merge_pcaps(pcaps_b, merged_out)
#             except Exception as e:
#                 print(f"  mergepcap error for {cls} from ScenarioB: {e}")
#                 merged_out = None

#         # 提取 sessions（第一次只用 ScenarioB 的合并结果）
#         tdir = os.path.join(tmp_root, f'{cls}_proc')
#         if os.path.exists(tdir):
#             shutil.rmtree(tdir)
#         os.makedirs(tdir, exist_ok=True)

#         temporal_list = []
#         mask_list = []
#         session_paths = []
#         if merged_out and os.path.exists(merged_out):
#             temporal_list, mask_list, session_paths, filtered, sessions_dir = gen_class_samples_from_merged_pcap(
#                 merged_out, tdir, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset,
#                 wave_name=wave_name, agg_points_num=agg_points)
#         else:
#             temporal_list, mask_list, session_paths, filtered, sessions_dir = ([], [], [], None, None)

#         # 如果不足 desired_per_class，则去 ScenarioA 找补充 pcap 后重新合并并再提取一次
#         if len(temporal_list) < desired_per_class:
#             print(f"  {cls} got {len(temporal_list)} sessions from ScenarioB, need {desired_per_class} -> searching ScenarioA ({scenario_a_dir}) ...")
#             pcaps_a = find_pcaps_for_class(cls, scenario_a_dir)
#             # 排除已经包含的同名文件
#             existing_paths = {os.path.abspath(p) for p in pcaps_b}
#             pcaps_a = [p for p in pcaps_a if os.path.abspath(p) not in existing_paths]
#             print(f"    ScenarioB pcaps: {len(pcaps_b)}, ScenarioA supplemental candidates after dedupe: {len(pcaps_a)}")

#             if pcaps_a:
#                 all_pcaps = (pcaps_b if pcaps_b else []) + pcaps_a
#                 # 重新合并到 merged_out（覆盖）
#                 try:
#                     if os.path.exists(merged_out):
#                         os.remove(merged_out)
#                 except Exception:
#                     pass
#                 try:
#                     merge_pcaps(all_pcaps, merged_out)
#                     # 重新生成（覆盖 tdir）
#                     if os.path.exists(tdir):
#                         shutil.rmtree(tdir)
#                     os.makedirs(tdir, exist_ok=True)
#                     temporal_list, mask_list, session_paths, filtered, sessions_dir = gen_class_samples_from_merged_pcap(
#                         merged_out, tdir, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset,
#                         wave_name=wave_name, agg_points_num=agg_points)
#                     print(f"  After supplement from ScenarioA, {cls} has {len(temporal_list)} sessions.")
#                 except Exception as e:
#                     print(f"  Failed to merge+extract for {cls} with ScenarioA supplements: {e}")
#             else:
#                 print(f"  No supplemental pcaps for {cls} found in ScenarioA.")

#         # 最后将该类的样本信息保存（可能仍小于 desired_per_class，后面会有警告）
#         class_samples[cls] = {
#             'temporal': temporal_list,
#             'mask': mask_list,
#             'session_paths': session_paths,
#             'merged_filtered': filtered if 'filtered' in locals() else None,
#             'sessions_dir': sessions_dir
#         }
#         print(f"  -> {cls}: extracted {len(temporal_list)} sessions (merged file: {merged_out})")

#     # 为每个 class 生成 temporal 数据（全部），然后根据 Benign 做采样
#     # class_samples = {}  # cls -> dict with temporal_list, mask_list, session_paths, filtered_pcap, sessions_dir
#     # for cls, merged_pcap in merged_cache.items():
#     #     print(f"[{time.strftime('%H:%M:%S')}] Generate sessions for class {cls} ...")
#     #     tdir = os.path.join(tmp_root, f'{cls}_proc')
#     #     if os.path.exists(tdir):
#     #         shutil.rmtree(tdir)
#     #     os.makedirs(tdir, exist_ok=True)
#     #     temporal_list, mask_list, session_paths, filtered, sessions_dir = gen_class_samples_from_merged_pcap(
#     #         merged_pcap, tdir, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset,
#     #         wave_name=wave_name, agg_points_num=agg_points)
#     #     print(f"  -> {cls}: {len(temporal_list)} sessions extracted")
#     #     class_samples[cls] = {
#     #         'temporal': temporal_list,
#     #         'mask': mask_list,
#     #         'session_paths': session_paths,
#     #         'merged_filtered': filtered,
#     #         'sessions_dir': sessions_dir
#     #     }

#     # 计算每类采样数：每个攻击类跟 Benign 保持一致，且 <= max_per_class
#     benign_count = len(class_samples['Benign']['temporal'])
#     print(f"Benign sessions: {benign_count}")
#     # 对每个 class，采样数 = min(len(class), benign_count, max_per_class)
#     per_class_counts = {}
#     for cls, info in class_samples.items():
#         cnt = min(len(info['temporal']), benign_count, max_per_class)
#         per_class_counts[cls] = cnt
#         print(f"  will sample {cnt} from class {cls} (available {len(info['temporal'])})")

#     # 随机采样并生成 contextual（针对采样的 session 子集）
#     for cls in sorted(class_samples.keys()):
#         info = class_samples[cls]
#         n_sample = per_class_counts[cls]
#         if n_sample == 0:
#             print(f"  skip class {cls} (0 samples)")
#             continue
#         idxs = list(range(len(info['temporal'])))
#         random.shuffle(idxs)
#         sel_idxs = idxs[:n_sample]
#         # collect temporal/mask
#         temporal_sel = [info['temporal'][i] for i in sel_idxs]
#         mask_sel = [info['mask'][i] for i in sel_idxs]
#         session_sel_paths = [info['session_paths'][i] for i in sel_idxs]
#         # build contextual for these sessions (uses merged_filtered pcap)
#         print(f"[{time.strftime('%H:%M:%S')}] Generating contextual for class {cls} ({n_sample} sessions)...")
#         contextual_arr = gen_contextual_for_sessions(info['merged_filtered'], session_sel_paths, wave_name=wave_name, agg_points=agg_points)
#         # sanity shapes
#         if contextual_arr.shape[0] != n_sample:
#             # 如果 contextual 生成数量不一致，尽量 align
#             min_n = min(contextual_arr.shape[0], n_sample)
#             temporal_sel = temporal_sel[:min_n]
#             mask_sel = mask_sel[:min_n]
#             session_sel_paths = session_sel_paths[:min_n]
#             contextual_arr = contextual_arr[:min_n]
#             n_sample = min_n

#         # append to global
#         all_temporal.extend(temporal_sel)
#         all_mask.extend(mask_sel)
#         all_contextual.append(contextual_arr)  # we'll vstack later
#         all_labels.extend([label_map[cls]] * n_sample)

#         print(f"  appended {n_sample} samples of class {cls}")

#     # concat and save
#     print("[*] Concatenating and saving...")
#     temporal_np = np.stack(all_temporal, axis=0).astype(np.int16)  # (N,64,64)
#     mask_np = np.stack(all_mask, axis=0).astype(np.uint8)
#     contextual_np = np.vstack(all_contextual).astype(np.float32) if len(all_contextual) > 0 else np.zeros((0,3,agg_points,agg_points), dtype=np.float32)
#     labels_np = np.array(all_labels, dtype=np.int32)

#     np.save(os.path.join(out_dir, 'temporal.npy'), temporal_np)
#     np.save(os.path.join(out_dir, 'mask.npy'), mask_np)
#     np.save(os.path.join(out_dir, 'contextual.npy'), contextual_np)
#     np.save(os.path.join(out_dir, 'labels.npy'), labels_np)
#     with open(os.path.join(out_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
#         json.dump(label_map, f, indent=2, ensure_ascii=False)

#     print(f"[DONE] Saved: temporal.npy ({temporal_np.shape}), mask.npy ({mask_np.shape}), contextual.npy ({contextual_np.shape}), labels.npy ({labels_np.shape})")
#     return {
#         'temporal_shape': temporal_np.shape,
#         'mask_shape': mask_np.shape,
#         'contextual_shape': contextual_np.shape,
#         'labels_shape': labels_np.shape,
#         'label_map': label_map
#     }

# # -----------------------
# # 主入口
# # -----------------------
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', type=str, default='./original_data/CrossNet2021_pcap/ScenarioB', help='IDS2017-2018 数据目录，包含 Benign 和 Attack 子目录')
#     parser.add_argument('--out_dir', type=str, default='./gene_data_all', help='输出目录')
#     parser.add_argument('--max_per_class', type=int, default=250, help='每个类的最大样本数（以控制运行时间）')
#     parser.add_argument('--tmp_root', type=str, default='./tmp_gen', help='临时文件目录')
#     parser.add_argument('--wave', type=str, default='cgau8', help='母 wavelet，默认 cgau8')
#     parser.add_argument('--session_len', type=int, default=64)
#     parser.add_argument('--packet_len', type=int, default=64)
#     parser.add_argument('--packet_offset', type=int, default=14)
#     parser.add_argument('--agg_points', type=int, default=128)
#     args = parser.parse_args()

#     random.seed(42)
#     np.random.seed(42)

#     t0 = time.time()
#     res = build_balanced_dataset(args.input_dir, args.out_dir, max_per_class=args.max_per_class,
#                                   session_len=args.session_len, packet_len=args.packet_len, packet_offset=args.packet_offset,
#                                   wave_name=args.wave, agg_points=args.agg_points, tmp_root=args.tmp_root)
#     t1 = time.time()
#     print("Total time: %.2f s" % (t1 - t0))
#     print(res)

import argparse
import os
import sys
import shutil
import subprocess
import time
import json
import binascii
import random

import numpy as np
import pandas as pd
import pywt
import joblib
import matplotlib
matplotlib.use('Agg')  # 不显示窗口
import matplotlib.pyplot as plt

import re

# -----------------------
# 参数与环境工具查找
# -----------------------
def find_wireshark_tools():
    wireshark_root = os.environ.get('WIRESHARK', None)
    candidates = []
    if wireshark_root:
        candidates.append(wireshark_root)
    candidates += [
        'C:\\Program Files\\Wireshark',
        'C:\\Program Files (x86)\\Wireshark',
        '/usr/bin',
        '/usr/local/bin'
    ]
    tools = {}
    for c in candidates:
        mc = os.path.join(c, 'mergecap.exe' if os.name == 'nt' else 'mergecap')
        tc = os.path.join(c, 'tshark.exe' if os.name == 'nt' else 'tshark')
        ec = os.path.join(c, 'editcap.exe' if os.name == 'nt' else 'editcap')
        if os.path.exists(mc) and 'mergecap' not in tools:
            tools['mergecap'] = mc
        if os.path.exists(tc) and 'tshark' not in tools:
            tools['tshark'] = tc
        if os.path.exists(ec) and 'editcap' not in tools:
            tools['editcap'] = ec
    # fallback to names (assume in PATH)
    tools.setdefault('mergecap', 'mergecap')
    tools.setdefault('tshark', 'tshark')
    tools.setdefault('editcap', 'editcap')
    return tools

TOOLS = find_wireshark_tools()
# SPLITCAP = os.path.join(os.getcwd(), "SplitCap.exe") if os.name == 'nt' else os.path.join(os.getcwd(), "SplitCap")  # assume available
SPLITCAP = "SplitCap.exe"
# -----------------------
# 辅助 I/O
# -----------------------
def list_pcaps_in_dir(d):
    exts = ('.pcap', '.pcapng')
    files = []
    for root, _, fnames in os.walk(d):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    files.sort()
    return files

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# -----------------------
# 基于你原始代码的 session pcap -> matrix 解析（保持逻辑）
# -----------------------
def parse_session_pcap_to_matrix(session_pcap_path, session_len=64, packet_len=64, packet_offset=14):
    """
    读取一个 session pcap 文件（raw bytes），返回 (session_matrix, padding_mask)
    如果 session 太短 (<3 packets)，返回 (None, None)
    """
    with open(session_pcap_path, 'rb') as f:
        content = f.read()
    hexc = binascii.hexlify(content)

    if len(hexc) < 48:
        return None, None

    # 判断小端顺序
    if hexc[:8] == b'd4c3b2a1':
        little_endian = True
    else:
        little_endian = False

    # 全局头 24 bytes -> 48 hex chars
    hexc = hexc[48:]

    packets_dec = []
    while len(hexc) > 0 and len(packets_dec) < session_len:
        if len(hexc) < 24:
            break
        frame_len = hexc[16:24]
        if little_endian:
            frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])
        try:
            frame_len = int(frame_len, 16)
        except Exception:
            break

        # remove current packet header (16 bytes -> 32 hex chars)
        hexc = hexc[32:]
        # get frame bytes, but only take packet_len bytes (with offset)
        frame_hex = hexc[packet_offset * 2: min(packet_len * 2, frame_len * 2)]
        frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)] if len(frame_hex) >= 2 else []
        packets_dec.append(frame_dec)

        # advance by whole frame_len (in hex length)
        hexc = hexc[frame_len * 2:]

    if len(packets_dec) < 3:
        return None, None

    # pad into matrix
    packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int32)
    session_matrix = np.ones((session_len, packet_len), dtype=np.int16) * -1
    row_idx = min(packets_dec_matrix.shape[0], session_len)
    col_idx = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

    # set irrelevant features to -1 (indices adapted by packet_offset)
    common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
    udp_irr_fea_idx = [40, 41]
    def sub(idx): return idx - packet_offset
    try:
        session_matrix[:, [sub(i) for i in common_irr_fea_idx]] = -1
    except Exception:
        pass
    # set tcp/udp-specific - using protocol field position 23 - packet_offset
    proto_col = 23 - packet_offset
    if 0 <= proto_col < session_matrix.shape[1]:
        # TCP
        for idx in tcp_irr_fea_idx:
            c = idx - packet_offset
            if 0 <= c < session_matrix.shape[1]:
                session_matrix[session_matrix[:, proto_col] == 6, c] = -1
        # UDP
        for idx in udp_irr_fea_idx:
            c = idx - packet_offset
            if 0 <= c < session_matrix.shape[1]:
                session_matrix[session_matrix[:, proto_col] == 17, c] = -1

    return session_matrix.astype(np.int16), (session_matrix == -1).astype(np.uint8)

# -----------------------
# Wavelet / contextual 特征（沿用 paper 中的思路）
# -----------------------
def wavelet_transform(seq, wave_name='cgau8', agg_points_num=128):
    """
    seq: 1D 数组长度 = agg_points_num（或小于，但会被 pad）
    返回 normalized spectrogram: shape (freqs, t) == (agg_points_num, agg_points_num) 但我们会截取合适部分
    """
    seq = np.array(seq, dtype=float)
    if seq.size < 1:
        seq = np.zeros(agg_points_num)
    # pad/truncate to agg_points_num
    if seq.shape[0] < agg_points_num:
        seq = np.pad(seq, (0, agg_points_num - seq.shape[0]), 'constant')
    else:
        seq = seq[:agg_points_num]

    scales = np.arange(1, agg_points_num + 1)
    try:
        fc = pywt.central_frequency(wave_name)
    except Exception:
        fc = pywt.central_frequency('cgau8')
    scales = 2 * fc * agg_points_num / scales
    try:
        cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)
    except Exception:
        cwtmatr, freqs = pywt.cwt(seq, scales, 'cgau8')
    spectrogram = np.log2((np.abs(cwtmatr)) ** 2 + 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram) + 1e-12)
    return spectrogram  # shape (freqs, t) where freqs == agg_points_num

# -----------------------
# 生成某个 class 的 temporal/mask/contextual（内存中返回）
# -----------------------

def gen_sessions_with_tshark(filtered_pcap, sessions_dir, session_len=64, packet_len=64, packet_offset=14):
    """
    使用 tshark 替代 SplitCap 进行会话分割
    """
    tshark = TOOLS['tshark']
    
    # 获取所有会话的五元组信息
    cmd = [tshark, '-r', filtered_pcap, '-Y', 'tcp or udp', '-T', 'fields', 
            '-e', 'frame.time_epoch', '-e', 'ip.src', '-e', 'ip.dst', 
            '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'udp.srcport', 
            '-e', 'udp.dstport', '-e', 'ipv6.src', '-e', 'ipv6.dst']
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark session analysis failed: {e}")

    # 解析会话信息
    sessions = {}
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 6:
            continue
            
        timestamp, src_ip, dst_ip, tcp_sport, tcp_dport, udp_sport, udp_dport = parts[:7]
        
        # 确定协议和端口
        if tcp_sport and tcp_dport:
            sport, dport, proto = tcp_sport, tcp_dport, 'tcp'
        elif udp_sport and udp_dport:
            sport, dport, proto = udp_sport, udp_dport, 'udp'
        else:
            continue
            
        # 构建会话键
        session_key = f"{src_ip}_{sport}_{dst_ip}_{dport}_{proto}"
        
        if session_key not in sessions:
            sessions[session_key] = {
                'src_ip': src_ip, 'dst_ip': dst_ip,
                'sport': sport, 'dport': dport, 'proto': proto
            }

    # 为每个会话提取数据包
    temporal_list = []
    mask_list = []
    used_session_paths = []
    
    for i, (session_key, session_info) in enumerate(sessions.items()):
        session_file = os.path.join(sessions_dir, f"session_{i}.pcap")
        
        # 构建显示过滤器
        if session_info['proto'] == 'tcp':
            display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and tcp.port=={session_info['sport']} and tcp.port=={session_info['dport']}"
        else:
            display_filter = f"ip.addr=={session_info['src_ip']} and ip.addr=={session_info['dst_ip']} and udp.port=={session_info['sport']} and udp.port=={session_info['dport']}"
        
        cmd = [tshark, '-r', filtered_pcap, '-Y', display_filter, '-w', session_file]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 解析会话文件
            sm, mask = parse_session_pcap_to_matrix(session_file, session_len, packet_len, packet_offset)
            if sm is not None:
                temporal_list.append(sm.astype(np.int16))
                mask_list.append(mask.astype(np.uint8))
                used_session_paths.append(session_file)
                
        except subprocess.CalledProcessError:
            continue

    return temporal_list, mask_list, used_session_paths, filtered_pcap, sessions_dir

def gen_class_samples_from_merged_pcap(merged_pcap_path, tmp_dir, session_len=64, packet_len=64, packet_offset=14,
                                      wave_name='cgau8', agg_points_num=128):
    """
    优化版本：直接使用 SplitCap，避免管道
    """
    ensure_dir(tmp_dir)
    base = os.path.splitext(os.path.basename(merged_pcap_path))[0]
    filtered = os.path.join(tmp_dir, f'{base}_filtered.pcap')
    sessions_dir = os.path.join(tmp_dir, f'{base}_sessions')
    
    if os.path.exists(sessions_dir):
        shutil.rmtree(sessions_dir)
    os.makedirs(sessions_dir, exist_ok=True)

    # 1) filter protocols with tshark
    display_filter = "not (arp or dhcp) and (tcp or udp)"
    tshark = TOOLS['tshark']
    cmd = [tshark, '-F', 'pcap', '-r', merged_pcap_path, '-w', filtered, '-Y', display_filter]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark filter error: {e.stderr.decode('utf-8', errors='ignore')}")

    # 2) 直接使用 SplitCap，避免管道
    splitcap = SPLITCAP
    cmd = [
        splitcap,
        "-r", filtered,  # 输入文件
        "-o", sessions_dir,       # 输出文件夹
        "-s", "session"                 # 按会话分割
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        # 如果 SplitCap 失败，尝试使用 tshark 替代方案
        print(f"SplitCap failed, trying tshark alternative: {e}")
        return gen_sessions_with_tshark(filtered, sessions_dir, session_len, packet_len, packet_offset)

    # 3) parse each session pcap into matrix
    session_pcaps = list_pcaps_in_dir(sessions_dir)
    temporal_list = []
    mask_list = []
    used_session_paths = []
    
    for sp in session_pcaps:
        sm, mask = parse_session_pcap_to_matrix(sp, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset)
        if sm is None:
            continue
        temporal_list.append(sm.astype(np.int16))
        mask_list.append(mask.astype(np.uint8))
        used_session_paths.append(sp)

    return temporal_list, mask_list, used_session_paths, filtered, sessions_dir

# -----------------------
# 生成 class 的 contextual 特征（只为选中的 session 列表生成 contextual）
# -----------------------
def gen_contextual_for_sessions(merged_pcap_path, selected_session_paths, wave_name='cgau8', agg_points=128):
    """
    使用 merged pcap 的 metadata，针对 selected_session_paths（list of session pcap）, 生成 contextual (N x 3 x 128 x 128)
    这里实现和原论文/代码思路一致的简化流程：
      - 从 merged_pcap_path 用 tshark 抽取 frame.time_epoch, frame.len, ip.src/dst, ipv6.src/dst, tcp/udp ports, tcp.flags.*
      - 为每个 session 从文件名解析出五元组，找到 session start time（最小 frame.time_epoch）
      - 对于每个 session，基于 start time 在全 pcap metadata 上按 ms/s/min 三个尺度聚合成 128 点序列 -> wavelet -> spectrogram
    返回: contextual_data (N, 3, 128, 128)
    """
    if len(selected_session_paths) == 0:
        return np.zeros((0, 3, agg_points, agg_points), dtype=np.float32)

    # 1) 用 tshark 抽取 csv 元数据
    tshark = TOOLS['tshark']
    csv_tmp = merged_pcap_path + '.meta.csv'
    fields = '-e frame.time_epoch -e frame.len -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst ' \
              '-e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport ' \
              '-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.fin'
    cmd = [tshark, '-T', 'fields'] + fields.split() + ['-r', merged_pcap_path, '-E', 'header=y', '-E', 'separator=,', '-E', 'occurrence=f']
    try:
        with open(csv_tmp, 'w', encoding='utf-8') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark metadata extract error: {e.stderr.decode('utf-8', errors='ignore')}")

    pcap_metadata = pd.read_csv(csv_tmp)
    # 补齐 src/dst/ip 与端口列 (与原代码逻辑类似，但更稳健)
    # 建立 src_ip/dst_ip 和 src_port/dst_port 列
    pcap_metadata['src_ip'] = pcap_metadata['ip.src'].fillna(pcap_metadata['ipv6.src'])
    pcap_metadata['dst_ip'] = pcap_metadata['ip.dst'].fillna(pcap_metadata['ipv6.dst'])
    # ports
    pcap_metadata['src_port'] = pcap_metadata['tcp.srcport'].fillna(pcap_metadata['udp.srcport'])
    pcap_metadata['dst_port'] = pcap_metadata['tcp.dstport'].fillna(pcap_metadata['udp.dstport'])
    # protocol label: if tcp.srcport not null -> TCP else UDP
    pcap_metadata['protocol'] = np.where(pcap_metadata['tcp.srcport'].notnull(), 'TCP', 'UDP')

    # helper to compute five_tuple_key same as original
    def make_five(row):
        try:
            a = str(row['src_ip'])
            b = str(int(float(row['src_port'])))
            c = str(row['dst_ip'])
            d = str(int(float(row['dst_port'])))
            proto = row['protocol']
            return '_'.join(sorted([a, b, c, d, proto]))
        except Exception:
            return ''
    pcap_metadata['frame.time_epoch'] = pcap_metadata['frame.time_epoch'].astype(float)
    pcap_metadata['frame.len'] = pd.to_numeric(pcap_metadata['frame.len'], errors='coerce').fillna(0).astype(float)
    pcap_metadata['five_tuple_key'] = pcap_metadata.apply(make_five, axis=1)

    # for each selected session, parse five-tuple from filename (SplitCap 格式: ???.protocol_src-dst_port... 需要健壮解析)
    contextual_list = []
    for sp in selected_session_paths:
        fname = os.path.basename(sp)
        # original code assumed session name contains ".protocol_srcport_dstip_dstport" style
        # We try to find a five-tuple key by searching pcap_metadata for same start time candidate
        # Fallback: compute earliest timestamp of entries in session pcap file itself via tshark
        try:
            # get earliest frame.time_epoch in this session pcap
            cmd = [TOOLS['tshark'], '-r', sp, '-T', 'fields', '-e', 'frame.time_epoch', '-c', '1']
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            st = res.stdout.strip().splitlines()
            if len(st) == 0 or st[0] == '':
                session_start_time = None
            else:
                session_start_time = float(st[0].strip())
        except Exception:
            session_start_time = None

        # find the closest frame in pcap_metadata
        if session_start_time is None:
            # fallback: use min frame.time_epoch across pcap
            session_start_time = pcap_metadata['frame.time_epoch'].min()

        # Build time keys for aggregation
        def build_agg_seq(agg_scale):
            # convert epoch -> time_key by dividing by agg_scale and int()
            time_key = (pcap_metadata['frame.time_epoch'] / agg_scale).map(int)
            center = int(session_start_time / agg_scale)
            start = center - agg_points // 2 + 1
            end = center + agg_points // 2
            sel = pcap_metadata[(time_key >= start) & (time_key <= end)]
            grouped = sel.groupby(time_key)['frame.len'].sum()
            agg = np.zeros(agg_points)
            for i, val in grouped.items():
                idx = int(i - start)
                if 0 <= idx < agg_points:
                    agg[idx] = val
            return agg

        ms_seq = build_agg_seq(0.001)
        s_seq = build_agg_seq(1)
        min_seq = build_agg_seq(60)

        ms_spec = wavelet_transform(ms_seq, wave_name=wave_name, agg_points_num=agg_points)
        s_spec = wavelet_transform(s_seq, wave_name=wave_name, agg_points_num=agg_points)
        min_spec = wavelet_transform(min_seq, wave_name=wave_name, agg_points_num=agg_points)

        contextual_list.append(np.stack([ms_spec, s_spec, min_spec], axis=0))  # (3,F,T)

    contextual_arr = np.array(contextual_list, dtype=np.float32)
    # cleanup csv_tmp to save space
    try:
        os.remove(csv_tmp)
    except Exception:
        pass
    return contextual_arr  # shape (N,3,128,128)

# -----------------------
# 合并 class 下所有 pcaps（若目录里有多个），返回一个合并后的临时 pcap 路径
# -----------------------
def merge_pcaps(pcaps, out_pcap_path):
    """
    pcaps: list of pcap file paths
    out_pcap_path: 输出文件路径
    """
    if len(pcaps) == 0:
        raise ValueError("no input pcaps to merge")
    if len(pcaps) == 1:
        shutil.copy2(pcaps[0], out_pcap_path)
        return out_pcap_path
    mergecap = TOOLS['mergecap']
    cmd = [mergecap, '-F', 'pcap', '-w', out_pcap_path] + pcaps
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mergecap error: {e.stderr.decode('utf-8', errors='ignore')}")
    return out_pcap_path

# -----------------------
# 主逻辑：遍历输入目录构建平衡数据集
# -----------------------
def build_balanced_dataset(input_dir, out_dir, max_per_class=2000,
                            session_len=64, packet_len=64, packet_offset=14,
                            wave_name='cgau8', agg_points=128, tmp_root='./tmp_gen'):
    """
    input_dir 假设包含 'Benign' 文件夹和 'Attack' 文件夹
    Attack 中可能是多个 pcap 文件或每类的子目录（比如 DDoS, Bot 等）
    输出: temporal.npy, mask.npy, contextual.npy, labels.npy, label_map.json 放到 out_dir
    """
    ensure_dir(out_dir)
    ensure_dir(tmp_root)

    # -----------------------
    # detect classes from filenames in input_dir
    # -----------------------
    # list all pcaps under input_dir (递归)
    pcaps = list_pcaps_in_dir(input_dir)
    if len(pcaps) == 0:
        raise RuntimeError(f"No pcap files found under {input_dir}")

    class_sources = {}
    for p in pcaps:
        b = os.path.basename(p)
        name = os.path.splitext(b)[0]
        # 抽取文件名前面直到第一个数字为止作为 class key
        m = re.match(r'^([^\d]+)', name)
        if m:
            cls = m.group(1).strip()
        else:
            # 如果没有前缀则用整个文件名（去掉扩展）
            cls = name.strip()
        if cls == '':
            cls = 'unknown'
        class_sources.setdefault(cls, []).append(p)

    print("Found classes:", list(class_sources.keys()))
    label_map = {c: i for i, c in enumerate(sorted(class_sources.keys()))}

    all_temporal = []
    all_mask = []
    all_contextual = []
    all_labels = []

    # merge each class's pcaps into 一个合并文件（缓存）
    merged_cache = {}
    for cls, pcaps in class_sources.items():
        print(f"[{time.strftime('%H:%M:%S')}] Merging class {cls}: {len(pcaps)} pcap(s)")
        merged_out = os.path.join(tmp_root, f'{cls}_merged.pcap')
        if os.path.exists(merged_out):
            os.remove(merged_out)
        merge_pcaps(pcaps, merged_out)
        merged_cache[cls] = merged_out

    # 为每个 class 生成 temporal 数据（全部）
    class_samples = {}
    for cls, merged_pcap in merged_cache.items():
        print(f"[{time.strftime('%H:%M:%S')}] Generate sessions for class {cls} ...")
        tdir = os.path.join(tmp_root, f'{cls}_proc')
        if os.path.exists(tdir):
            shutil.rmtree(tdir)
        os.makedirs(tdir, exist_ok=True)
        temporal_list, mask_list, session_paths, filtered, sessions_dir = gen_class_samples_from_merged_pcap(
            merged_pcap, tdir, session_len=session_len, packet_len=packet_len, packet_offset=packet_offset,
            wave_name=wave_name, agg_points_num=agg_points)
        print(f"  -> {cls}: {len(temporal_list)} sessions extracted")
        class_samples[cls] = {
            'temporal': temporal_list,
            'mask': mask_list,
            'session_paths': session_paths,
            'merged_filtered': filtered,
            'sessions_dir': sessions_dir
        }

    # -----------------------
    # 设定每类目标样本数（用户要求每类提取 400 个）
    # -----------------------
    target_per_class = 400
    per_class_counts = {}
    for cls, info in class_samples.items():
        cnt = min(len(info['temporal']), target_per_class)
        per_class_counts[cls] = cnt
        print(f"  will sample {cnt} from class {cls} (available {len(info['temporal'])})")

    # 随机采样并生成 contextual（针对采样的 session 子集）
    for cls in sorted(class_samples.keys()):
        info = class_samples[cls]
        n_sample = per_class_counts[cls]
        if n_sample == 0:
            print(f"  skip class {cls} (0 samples)")
            continue
        idxs = list(range(len(info['temporal'])))
        random.shuffle(idxs)
        sel_idxs = idxs[:n_sample]
        temporal_sel = [info['temporal'][i] for i in sel_idxs]
        mask_sel = [info['mask'][i] for i in sel_idxs]
        session_sel_paths = [info['session_paths'][i] for i in sel_idxs]

        print(f"[{time.strftime('%H:%M:%S')}] Generating contextual for class {cls} ({n_sample} sessions)...")
        contextual_arr = gen_contextual_for_sessions(info['merged_filtered'], session_sel_paths, wave_name=wave_name, agg_points=agg_points)
        # 如果 contextual 数量不一致，align 至最小值
        if contextual_arr.shape[0] != n_sample:
            min_n = min(contextual_arr.shape[0], n_sample)
            temporal_sel = temporal_sel[:min_n]
            mask_sel = mask_sel[:min_n]
            session_sel_paths = session_sel_paths[:min_n]
            contextual_arr = contextual_arr[:min_n]
            n_sample = min_n

        all_temporal.extend(temporal_sel)
        all_mask.extend(mask_sel)
        all_contextual.append(contextual_arr)
        all_labels.extend([label_map[cls]] * n_sample)

        print(f"  appended {n_sample} samples of class {cls}")

    # concat and save
    print("[*] Concatenating and saving...")
    if len(all_temporal) == 0:
        raise RuntimeError("No sessions were extracted for any class.")

    temporal_np = np.stack(all_temporal, axis=0).astype(np.int16)  # (N,64,64)
    mask_np = np.stack(all_mask, axis=0).astype(np.uint8)
    contextual_np = np.vstack(all_contextual).astype(np.float32) if len(all_contextual) > 0 else np.zeros((0,3,agg_points,agg_points), dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.int32)

    np.save(os.path.join(out_dir, 'temporal.npy'), temporal_np)
    np.save(os.path.join(out_dir, 'mask.npy'), mask_np)
    np.save(os.path.join(out_dir, 'contextual.npy'), contextual_np)
    np.save(os.path.join(out_dir, 'labels.npy'), labels_np)
    with open(os.path.join(out_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved: temporal.npy ({temporal_np.shape}), mask.npy ({mask_np.shape}), contextual.npy ({contextual_np.shape}), labels.npy ({labels_np.shape})")
    return {
        'temporal_shape': temporal_np.shape,
        'mask_shape': mask_np.shape,
        'contextual_shape': contextual_np.shape,
        'labels_shape': labels_np.shape,
        'label_map': label_map
    }

# -----------------------
# 主入口
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./original_data/VPN2016/NonVPN', help='IDS2017-2018 数据目录，包含 Benign 和 Attack 子目录')
    parser.add_argument('--out_dir', type=str, default='./gene_data_all', help='输出目录')
    parser.add_argument('--max_per_class', type=int, default=100, help='每个类的最大样本数（以控制运行时间）')
    parser.add_argument('--tmp_root', type=str, default='./tmp_gen', help='临时文件目录')
    parser.add_argument('--wave', type=str, default='cgau8', help='母 wavelet，默认 cgau8')
    parser.add_argument('--session_len', type=int, default=64)
    parser.add_argument('--packet_len', type=int, default=64)
    parser.add_argument('--packet_offset', type=int, default=14)
    parser.add_argument('--agg_points', type=int, default=128)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    res = build_balanced_dataset(args.input_dir, args.out_dir, max_per_class=args.max_per_class,
                                  session_len=args.session_len, packet_len=args.packet_len, packet_offset=args.packet_offset,
                                  wave_name=args.wave, agg_points=args.agg_points, tmp_root=args.tmp_root)
    t1 = time.time()
    print("Total time: %.2f s" % (t1 - t0))
    print(res)
