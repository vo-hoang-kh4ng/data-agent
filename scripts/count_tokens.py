import os
import glob

def estimate_tokens(directory):
    md_files = glob.glob(os.path.join(directory, "*.md"))
    total_input_chars = 0
    total_output_chars = 0
    
    for fpath in md_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by "Input: " and "Output: "
        # We can just find the indices.
        parts = []
        i = 0
        while i < len(content):
            in_idx = content.find("Input: ", i)
            out_idx = content.find("Output: ", i)
            
            if in_idx == -1:
                break
                
            if out_idx == -1:
                # Only input left
                parts.append(("input", content[in_idx:]))
                break
                
            if in_idx < out_idx:
                parts.append(("input", content[in_idx:out_idx]))
                i = out_idx
            else:
                parts.append(("output", content[out_idx:in_idx]))
                i = in_idx

        # Calculate quadratic sum
        history_chars = 0
        for ptype, text in parts:
            chars = len(text)
            if ptype == "input":
                history_chars += chars
                total_input_chars += history_chars
            elif ptype == "output":
                total_output_chars += chars
                history_chars += chars
                
    # Approximate tokens = chars / 4
    return total_input_chars // 4, total_output_chars // 4

if __name__ == "__main__":
    d = r"d:\kh4ng\Data_agent\dgm_agent\swe_bench\predictions_100\hosted_vllm\Qwen\Qwen3.5-35B-A3B-FP8_0"
    inp, out = estimate_tokens(d)
    print(f"Estimated Input Tokens: {inp:,}")
    print(f"Estimated Output Tokens: {out:,}")
    print(f"Estimated Total Tokens Burned: {inp + out:,}")
