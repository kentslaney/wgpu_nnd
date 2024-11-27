.PHONY: debug

debug: tmp.npy
	cargo run tmp.npy 5 1

tmp.npy: .venv
	.venv/bin/python -c "import numpy as np; np.random.seed(0); \
		np.save('tmp.npy', np.float32(np.random.uniform(size=(8, 4))))"

.venv:
	@if ! which python >/dev/null; then exit 1; fi
	if [ ! -d .venv ]; then python -m venv .venv; fi
	if [ ! -d .venv/lib/*/site-packages/numpy ]; \
		then .venv/bin/pip install -U numpy; fi
