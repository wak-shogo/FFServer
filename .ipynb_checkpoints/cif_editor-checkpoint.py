# cif_editor.py
import random
import sys
from io import StringIO

def modify_cif_content(input_content, replace_from='Bi', replace_to='Mg', percentage=0.2):
    """
    CIFファイルの内容（文字列）を受け取り、指定された元素を指定された割合で置き換えた
    新しいCIFファイルの内容（文字列）とログメッセージを返します。
    """
    lines = input_content.splitlines(keepends=True)
    log_messages = []

    new_lines = lines[:]
    atom_site_loop_indices = []
    in_atom_site_loop = False
    header_map = {}
    header_order = []
    data_start_line = -1

    # _atom_site_ loopを探索
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip.startswith('loop_'):
            j = i + 1
            potential_headers = []
            potential_header_indices = {}
            while j < len(lines) and lines[j].strip().startswith('_atom_site_'):
                header = lines[j].strip()
                potential_headers.append(header)
                potential_header_indices[header] = len(potential_headers) - 1
                j += 1
            
            if '_atom_site_label' in potential_header_indices:
                in_atom_site_loop = True
                header_map = potential_header_indices
                header_order = potential_headers
                data_start_line = j
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('loop_', '_', '#', 'data_')):
                    if len(lines[j].split()) >= len(header_order):
                        atom_site_loop_indices.append(j)
                    j += 1
                break
    
    if not in_atom_site_loop:
        log_messages.append("Error: Could not find a valid _atom_site_ loop in the CIF file.")
        return "".join(lines), log_messages

    label_col_index = header_map['_atom_site_label']
    symbol_col_index = header_map.get('_atom_site_type_symbol', -1)

    candidate_indices = []
    for idx in atom_site_loop_indices:
        parts = lines[idx].split()
        if len(parts) > label_col_index and parts[label_col_index].startswith(replace_from):
            candidate_indices.append(idx)

    if not candidate_indices:
        log_messages.append(f"Info: No atoms with site label starting with '{replace_from}' were found.")
        return "".join(lines), log_messages

    num_to_replace = int(len(candidate_indices) * percentage)
    indices_to_replace = random.sample(candidate_indices, num_to_replace)

    replaced_count = 0
    for line_idx in indices_to_replace:
        original_line = lines[line_idx]
        parts = original_line.split()
        
        if len(parts) < max(label_col_index, symbol_col_index) + 1:
            log_messages.append(f"Warning: Skipping line {line_idx+1} due to insufficient columns.")
            continue
        
        # サイトラベルの置換
        original_label = parts[label_col_index]
        parts[label_col_index] = replace_to + original_label[len(replace_from):]
        
        # 元素タイプの置換
        if symbol_col_index != -1 and parts[symbol_col_index] == replace_from:
            parts[symbol_col_index] = replace_to

        # スペースを維持して行を再構築
        new_line_segments = []
        current_part_idx = 0
        current_char_idx = 0
        original_line_len = len(original_line.rstrip('\n'))
        while current_char_idx < original_line_len:
            char = original_line[current_char_idx]
            if char.isspace():
                start_space = current_char_idx
                while current_char_idx < original_line_len and original_line[current_char_idx].isspace():
                    current_char_idx += 1
                new_line_segments.append(original_line[start_space:current_char_idx])
            else:
                if current_part_idx < len(parts):
                    new_line_segments.append(parts[current_part_idx])
                    start_part = current_char_idx
                    while current_char_idx < original_line_len and not original_line[current_char_idx].isspace():
                        current_char_idx += 1
                    current_part_idx += 1
                else:
                    new_line_segments.append(original_line[current_char_idx:])
                    current_char_idx = original_line_len
        
        new_lines[line_idx] = "".join(new_line_segments) + '\n'
        replaced_count += 1
    
    log_messages.append(f"Success: Replaced {replaced_count} out of {len(candidate_indices)} '{replace_from}' sites with '{replace_to}'.")
    return "".join(new_lines), log_messages