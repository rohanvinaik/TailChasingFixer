# Tail-Chasing Detector VS Code Extension

Real-time detection of LLM-assisted tail-chasing anti-patterns in your Python code.

## Features

- 🔍 **Real-time Analysis**: Detect tail-chasing patterns as you type
- 🧠 **Semantic Duplicate Detection**: Find functions with same behavior but different implementations
- 📊 **Inline Diagnostics**: See issues directly in your code
- 💡 **Smart Suggestions**: Get actionable fixes for detected patterns
- 📈 **Code Lens**: Visual indicators for semantic duplicates

## Installation

1. Install the Python package:
   ```bash
   pip install tail-chasing-detector
   ```

2. Install the VS Code extension:
   - Open VS Code
   - Go to Extensions (Cmd+Shift+X)
   - Search for "Tail-Chasing Detector"
   - Click Install

## Usage

### Commands

- `Tail-Chasing: Analyze` - Analyze current file
- `Tail-Chasing: Show Semantic Duplicates` - View all semantic duplicates
- `Tail-Chasing: Toggle Real-Time Analysis` - Enable/disable as-you-type analysis

### Configuration

```json
{
  "tailchasing.enable": true,
  "tailchasing.semantic.enable": true,
  "tailchasing.semantic.hvDim": 8192,
  "tailchasing.realTime": false,
  "tailchasing.severity": "Warning"
}
```

## Examples

The extension will highlight:

1. **Semantic Duplicates**:
   ```python
   def calculate_average(numbers):  # 🔄 Semantic duplicate of compute_mean
       return sum(numbers) / len(numbers)
   
   def compute_mean(values):
       total = 0
       for v in values:
           total += v
       return total / len(values)
   ```

2. **Phantom Functions**:
   ```python
   def process_data():  # ⚠️ Empty placeholder function
       pass
   ```

3. **Circular Imports**:
   ```python
   from module_b import helper  # ❌ Circular import detected
   ```

## Requirements

- VS Code 1.74.0 or higher
- Python 3.9 or higher
- tail-chasing-detector package installed

## License

MIT