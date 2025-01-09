# Multi-task Knowledge Sharing Analysis

As a supplementary material, this visualization demonstrates the empirical evidence of structural similarities shared across different robot design tasks in the MISCO framework.

## Overview
The figure consists of three key components analyzing the shared structures and knowledge transfer across multiple robotic tasks:

<p align="center">
  <img src="6-multi-task intuitive.png" width="700">
  <br>
  <em>Figure: Shared structures across tasks. a, Overlapping of robot designs in multiple tasks visualized using t-SNE, where the red boxes highlight shared morphological features across different tasks. b, Perplexity matrix showing the similarity between different tasks, where values close to 1 (yellow) indicate high similarity. Note that 'DownStepper-v0' is shortened as 'DS-v0' and 'PlatformJumper-v0' as 'PJ-v0'. c, Dimension-wise average KL divergence of task latent variables, where bubble size and color intensity correspond to divergence magnitude across 128 dimensions, revealing the balance between shared and task-specific information in the latent space.</em>
</p>

## Technical Details

### Data Source
- Elite morphological samples from top 5% of 1,000 robots evaluations
- Three separate single-task learning experiments: GA, BO, and CPPN-NEAT
- 450 elite samples per task from three repeated experiments

### Analysis Methods
1. **Qualitative Analysis (Panel a)**
   - t-SNE dimensionality reduction
   - Visualization of shared morphological features
   - Red boxes indicate overlapping design spaces

2. **Perplexity Analysis (Panel b)**
   - Measures task similarity through intra-class and inter-class distances
   - Values close to 1 indicate high similarity between tasks
   - Demonstrates both task independence and shared features

3. **KL Divergence Analysis (Panel c)**
   - 128-dimensional Gaussian distribution analysis
   - Smaller bubbles indicate shared information across tasks
   - Larger bubbles represent task-specific variations

## Key Findings
- Evidence of shared morphological features across different tasks
- Balance between task-specific and shared structural elements
- Quantitative support for multi-task knowledge sharing in robot design
- Validation of MISCO's approach to multi-task optimization

This analysis supports the effectiveness of multi-task learning within MISCO, demonstrating its ability to leverage shared knowledge while maintaining task-specific optimization capabilities.

