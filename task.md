Please implement a Python function for a **fixed-size grid partition sampler**.

### Goal

Partition the space into a **uniform grid with fixed cell size**. Then build a sampler that:


2. Draws a **fixed number of samples from each cell**.
3. Uses those sampled points to compute a loss.
4. Applies **per-sample weights** during loss computation, where the weight for samples from a given cell is **proportional to the average loss of the samples in that same cell**.

### Requirements

* Implement the grid partitioning using **fixed-size cells** only.
* From each non-empty cell, sample exactly `k` points (or make this configurable).
* The loss for each sampled point should be multiplied by a weight.
* The weight assigned to samples from a cell should be based on the **mean loss of all sampled points in that cell**.
* Make the implementation clean, reusable, and well-documented.

### Notes

* If a cell contains fewer than `k` points, handle it gracefully, for example by sampling with replacement or by raising a clear error
* Follow other sampler style.
* If helpful, structure the code in a PyTorch-friendly way.
