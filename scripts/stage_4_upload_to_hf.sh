#!/bin/bash
huggingface-cli upload mateuszgrzyb/lichess-stockfish-normalized ./data/stage_3_split/ --repo-type=dataset
# README:
# If you want to upload dataset do HD, you need to:
#   1. Createre HF account.
#   2. Create and copy access token: https://huggingface.co/settings/tokens
#   3. Login through CLI: `huggingface-cli login`.
#   4. Change file permission: `chmod +x upload_to_hf.sh` 
#   5. Create your dataset in HF.
#   6. Replace "mateuszgrzyb/lichess-stockfish-normalized" with address of your dataset.
#   7. Run this file: `./upload_to_hf.sh`.