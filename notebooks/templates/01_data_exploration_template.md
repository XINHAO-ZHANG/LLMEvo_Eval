# Data Exploration Notebook Template

This notebook template provides a starting point for exploring your optimization task data.

## Setup

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tasks import get_task

# Configure plots
plt.style.use('default')
sns.set_palette("husl")
```

## Load Task Data

```python
# Load your task
task_name = "tsp"  # Change this to your task
task_module = get_task(task_name)

print(f"Task: {task_name}")
print(f"Description: {getattr(task_module, '__doc__', 'No description')}")
```

## Data Analysis

Add your data analysis code here...

```python
# Example: Load and visualize data
# data = task_module.load_data("../data/your_dataset")
# ...
```

## Visualization

Create visualizations of your data...

```python
# Example plotting code
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Add your plots here...
plt.tight_layout()
plt.show()
```