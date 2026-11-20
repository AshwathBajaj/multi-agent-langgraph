# SciFact sample

This directory is a small subset of the public SciFact release, kept so the
tests and the default demo run without a download. It is not covered by the
MIT license on the rest of the repository.

- Dataset: SciFact (Wadden, Lin, Lo, Wang, van Zuylen, Cohan, Hajishirzi, EMNLP 2020)
- Paper: https://aclanthology.org/2020.emnlp-main.609/
- Download used: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
- Project: https://github.com/allenai/scifact
- Split: dev claims that have at least one evidence abstract
- What "relevant" means: the abstract is listed as evidence for that claim
- Corpus origin: S2ORC abstracts shipped inside the SciFact release
- GitHub license on allenai/scifact: Other (no SPDX identifier)
- S2ORC today: ODC-By 1.0 (https://opendatacommons.org/licenses/by/1-0/). The original S2ORC release was non-commercial. SciFact is a 2020 snapshot of that corpus, republished by AllenAI for the claim-verification task.

The vectors in `vectors.json` were computed locally with
`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) and are not part of SciFact.

Do not treat these abstracts as MIT-licensed code. `scripts/download_scifact.py`
fetches the full release into `data/cache/` and can rebuild this sample.
