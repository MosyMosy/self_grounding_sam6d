import sys
import os
import torch
import time
import csv
from ptflops import get_model_complexity_info

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from model.DinoV2_modified.vision_transformer import (
    vit_small as vit_small_g,
    vit_base as vit_base_g,
    vit_large as vit_large_g,
    vit_giant2 as vit_giant2_g,
)


# Function to measure inference time and memory usage
def measure_performance(model_fn, input_size=(10, 3, 224, 224), device="cuda"):
    # Clean up any previous memory allocations
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    # Instantiate and move the model to the device
    model = model_fn(block_chunks=0)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Reset peak memory stats before measurement
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated(device=device)
    else:
        memory_before = 0

    # Measure inference time and peak memory during forward pass
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device=device)
        memory_after = torch.cuda.memory_allocated(device=device)
        peak_memory_during_forward = peak_memory - memory_before
        peak_memory_during_forward_MB = peak_memory_during_forward / (1024**2)
    else:
        peak_memory_during_forward_MB = 0

    # Clean up
    del model
    del dummy_input
    if device == "cuda":
        torch.cuda.empty_cache()

    return inference_time, peak_memory_during_forward_MB


# Initialize TSV file with tab delimiter
tsv_file = "model_performance.tsv"  # Changed extension to .tsv for clarity
fieldnames = [
    "Model",
    "Parameters",
    "GFLOPs",
    "Inference Time (ms)",
    "Peak Memory (MB)",
]


def vit_modified(measure_performance, writer):
    # Second loop for the modified DINOv2 models
    model_functions = [vit_small_g, vit_base_g, vit_large_g, vit_giant2_g]
    model_names = ["vit_small_g", "vit_base_g", "vit_large_g", "vit_giant2_g"]

    for model_fn, name in zip(model_functions, model_names):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Compute GFLOPs and number of parameters
        model = model_fn(block_chunks=0)
        model.to(device)
        model.eval()
        macs, params = get_model_complexity_info(
            model,
            (3, 224, 224),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        macs_value = float(macs.replace("GMac", "").strip())
        gflops = macs_value * 2  # Since 1 MAC ≈ 2 FLOPs

        # Clean up model before performance measurement
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Measure inference time and memory usage
        inference_time, peak_memory = measure_performance(model_fn, device=device)

        # Display results
        print(f"\nModel: {name}")
        print(f"Number of parameters: {params}")
        print(f"Computational complexity (GFLOPs): {gflops}")
        print(f"Inference Time: {inference_time:.2f} ms")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")

        # Write results to TSV
        writer.writerow(
            {
                "Model": name,
                "Parameters": params,
                "GFLOPs": gflops,
                "Inference Time (ms)": f"{inference_time:.2f}",
                "Peak Memory (MB)": f"{peak_memory:.2f}",
            }
        )


def vit_main(measure_performance, writer):
    # First loop for the standard ViT models
    model_functions = [vit_small, vit_base, vit_large, vit_giant2]
    model_names = ["vit_small", "vit_base", "vit_large", "vit_giant2"]
    for model_fn, name in zip(model_functions, model_names):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Compute GFLOPs and number of parameters
        model = model_fn(block_chunks=0)
        model.to(device)
        model.eval()
        macs, params = get_model_complexity_info(
            model,
            (3, 224, 224),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        macs_value = float(macs.replace("GMac", "").strip())
        gflops = macs_value * 2  # Since 1 MAC ≈ 2 FLOPs

        # Clean up model before performance measurement
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        # Measure inference time and memory usage
        inference_time, peak_memory = measure_performance(model_fn, device=device)

        # Display results
        print(f"\nModel: {name}")
        print(f"Number of parameters: {params}")
        print(f"Computational complexity (GFLOPs): {gflops}")
        print(f"Inference Time: {inference_time:.2f} ms")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")

        # Write results to TSV
        writer.writerow(
            {
                "Model": name,
                "Parameters": params,
                "GFLOPs": gflops,
                "Inference Time (ms)": f"{inference_time:.2f}",
                "Peak Memory (MB)": f"{peak_memory:.2f}",
            }
        )


with open(tsv_file, "w", newline="") as tsvfile:
    writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    
    for i in range(10):

        print("\nModified DINOv2 Models:")
        vit_modified(measure_performance, writer)

        print("Standard ViT Models:")
        vit_main(measure_performance, writer)


print(f"\nPerformance metrics have been saved to '{tsv_file}'.")
