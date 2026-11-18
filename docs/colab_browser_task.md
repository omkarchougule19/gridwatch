# Task: run the GridWatch Chronos-2 fine-tuning on Google Colab (no Google Drive)

You are driving my browser. I'm signed in to Google. Stop and ask me if a step fails twice, or if a file picker, permission popup, or pay/upgrade prompt appears.

Both files are on my Desktop (`<your Desktop>\`):
- `chronos_colab.ipynb`
- `gridwatch_colab.zip`

## Steps
1. Open https://colab.research.google.com → **File → Upload notebook** → `chronos_colab.ipynb`. I may need to pick the file myself.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the first code cell. A **Choose Files** button appears: I pick `gridwatch_colab.zip`. Wait for the unzip and `pip install` to finish (about 2–3 min).
4. Run the next two code cells (load data, define `run_and_save`).
5. **Skip** the code cell under "1. Zero-shot".
6. Run the code cell under "2. Fine-tuned" (about 10–20 min). Don't re-run it while it's still going.
7. Run the "3. Quick look" cell, then the last cell, which downloads two `.pkl` files to my Downloads (allow multiple downloads if the browser asks).

## On errors
- Copy the last 30 lines of the error and stop.
- **CUDA out of memory:** change `batch_size=32` to `batch_size=16` in the fine-tune cell and re-run only that cell.
